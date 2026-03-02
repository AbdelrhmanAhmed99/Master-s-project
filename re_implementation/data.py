"""
data.py — Multilingual NMT data pipeline  (many → English).

Loads 5 language pairs from 4 corpus families, each targeting English:
    de → en   (German)      fr → en   (French)     cs → en   (Czech)
    ru → en   (Russian)     es → en   (Spanish)

Data sources and their on-disk formats:
    1. Europarl v10           TSV.GZ   col[0]=src  col[1]=en  [+ meta cols]
    2. News Commentary v18    TSV.GZ   <lang>-en → col[0]=src, col[1]=en
                                       en-<lang> → col[0]=en,  col[1]=src  (SWAPPED)
    3. WikiMatrix v1          TSV.GZ   col[0]=score  col[1]=lang1  col[2]=lang2
                                       col[3]=lid1  col[4]=lid2
                                       en-<lang> → col[1]=en,  col[2]=src  (SWAPPED)
    4. Tilde Model Corpus     TMX XML  segment language from xml:lang attribute

Splitting strategy — 3-way stratified by (language, dataset):
    train  ~98 %        val  ~1 %        test  ~1 %
    The test split is sampled from the existing corpus and saved as
    fixed reusable TSV files (one per language pair) so that subsequent
    runs always evaluate on exactly the same sentences.
    Each test TSV row: src <TAB> en <TAB> pair <TAB> dataset
"""

import gzip
import hashlib
import html
import multiprocessing as mp
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ── Special token IDs (match SentencePiece defaults) ──────────
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2   # <s>  / bos
EOS_ID = 3   # </s> / eos

# ── Languages ─────────────────────────────────────────────────
LANG_PAIRS: List[str] = ["de", "fr", "cs", "ru", "es"]

# (src_text, tgt_en_text, lang_code, dataset_name)
DataRecord = Tuple[str, str, str, str]


# ==============================================================
# SentencePiece wrapper
# ==============================================================

class SPTokenizer:
    """Thin wrapper around a SentencePiece model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list:
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts (uses SP's internal threading)."""
        return [self.sp.encode(t, out_type=int) for t in texts]

    def decode_ids(self, ids) -> str:
        clean = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self.sp.decode(clean)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)


def _train_sentencepiece(
    text_paths: List[str],
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    num_threads: int = 16,
    input_sentence_size: int = 5_000_000,
):
    """Train a shared SentencePiece BPE model on multiple text files."""
    print(f"[data] Training SentencePiece ({model_type}, vocab={vocab_size}) "
          f"on {len(text_paths)} files …")
    spm.SentencePieceTrainer.Train(
        input=",".join(text_paths),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        train_extremely_large_corpus=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=SOS_ID,
        eos_id=EOS_ID,
        num_threads=num_threads,
        # 0.9999 covers Latin + Cyrillic scripts fully
        character_coverage=0.9999,
    )
    print(f"[data] SentencePiece model saved → {model_prefix}.model")


# ==============================================================
# Utility: read plain-text file
# ==============================================================

def read_lines(path: str) -> list:
    """Read a text file, one line per element, strip blanks."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ==============================================================
# Dataset / DataLoader  (unchanged interface)
# ==============================================================

class TranslationDataset(torch.utils.data.Dataset):
    """Parallel sentence pairs stored as pre-encoded integer-ID lists."""

    def __init__(self, src_ids, tgt_ids):
        assert len(src_ids) == len(tgt_ids)
        self.src = src_ids
        self.tgt = tgt_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch, pad_id: int = PAD_ID):
    """Collate (src_ids, tgt_ids) pairs into padded tensors.

    Returns
    -------
    src         (batch, src_len)
    src_lengths (batch,)
    tgt_in      (batch, tgt_len)  — <s> + tokens
    tgt_out     (batch, tgt_len)  — tokens + </s>
    """
    src_ids, tgt_ids = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long)
    src = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in src_ids],
        batch_first=True, padding_value=pad_id,
    )

    tgt_in_list = [[SOS_ID] + list(t) for t in tgt_ids]
    tgt_out_list = [list(t) + [EOS_ID] for t in tgt_ids]

    tgt_in = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgt_in_list],
        batch_first=True, padding_value=pad_id,
    )
    tgt_out = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgt_out_list],
        batch_first=True, padding_value=pad_id,
    )

    return src, src_lengths, tgt_in, tgt_out


def get_dataloader(dataset: TranslationDataset, batch_size: int,
                   shuffle: bool = True, num_workers: int = 2):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


# ==============================================================
# Raw corpus readers  (one function per corpus family)
# ==============================================================

def _unescape(text: str) -> str:
    """Decode HTML entities (common in TMX) and strip whitespace."""
    return html.unescape(text).strip()


def _read_tsv_gz(path: str) -> List[Tuple[str, str]]:
    """Return (col0, col1) from a gzip TSV, skipping bad rows."""
    pairs = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                pairs.append((a, b))
    return pairs


# ------ Europarl v10 --------------------------------------------------

def _read_europarl(root: str) -> List[DataRecord]:
    """Europarl v10: col[0]=src  col[1]=en (always <lang>-en naming)."""
    d = os.path.join(root, "europarl-v10")
    if not os.path.isdir(d):
        return []
    records: List[DataRecord] = []
    for lang in LANG_PAIRS:
        path = os.path.join(d, f"europarl-v10.{lang}-en.tsv.gz")
        if not os.path.isfile(path):
            continue
        pairs = _read_tsv_gz(path)
        for src, en in pairs:
            records.append((src, en, lang, "europarl_v10"))
        print(f"[data]   europarl_v10  {lang}-en : {len(pairs):>10,}")
    return records


# ------ News Commentary v18 -------------------------------------------

def _read_news_commentary(root: str) -> List[DataRecord]:
    """News Commentary v18.  Two naming styles:
       <lang>-en.tsv.gz  → col[0]=src,  col[1]=en   (normal)
       en-<lang>.tsv.gz  → col[0]=EN,   col[1]=src  (SWAPPED — flip)
    """
    d = os.path.join(root, "news_commentary_v18_en")
    if not os.path.isdir(d):
        return []
    records: List[DataRecord] = []
    for lang in LANG_PAIRS:
        for fname, swap in [
            (f"news-commentary-v18.{lang}-en.tsv.gz", False),
            (f"news-commentary-v18.en-{lang}.tsv.gz", True),
        ]:
            path = os.path.join(d, fname)
            if not os.path.isfile(path):
                continue
            pairs = _read_tsv_gz(path)
            for c0, c1 in pairs:
                src, en = (c1, c0) if swap else (c0, c1)
                records.append((src, en, lang, "news_commentary_v18"))
            print(f"[data]   news_commentary_v18  {lang}-en : "
                  f"{len(pairs):>10,}  (swap={swap})")
    return records


# ------ WikiMatrix v1 -------------------------------------------------

def _read_wikimatrix(root: str, min_score: float = 1.05) -> List[DataRecord]:
    """WikiMatrix: score <TAB> lang1_text <TAB> lang2_text <TAB> lid1 <TAB> lid2.
    <lang>-en → col[1]=src,  col[2]=en   (normal)
    en-<lang> → col[1]=EN,   col[2]=src  (SWAPPED — flip)
    Pairs below *min_score* are dropped (noise filter).
    """
    d = os.path.join(root, "wikimatrix_en_pairs")
    if not os.path.isdir(d):
        return []
    records: List[DataRecord] = []
    for lang in LANG_PAIRS:
        for fname, swap in [
            (f"WikiMatrix.v1.{lang}-en.langid.tsv.gz", False),
            (f"WikiMatrix.v1.en-{lang}.langid.tsv.gz", True),
        ]:
            path = os.path.join(d, fname)
            if not os.path.isfile(path):
                continue
            kept = 0
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        score = float(parts[0])
                    except ValueError:
                        continue
                    if score < min_score:
                        continue
                    c1, c2 = parts[1].strip(), parts[2].strip()
                    if not c1 or not c2:
                        continue
                    src, en = (c2, c1) if swap else (c1, c2)
                    records.append((src, en, lang, "wikimatrix_v1"))
                    kept += 1
            print(f"[data]   wikimatrix_v1  {lang}-en : "
                  f"{kept:>10,}  (score≥{min_score}, swap={swap})")
    return records


# ------ Tilde Model Corpus (TMX) --------------------------------------

def _parse_tmx(path: str, src_lang: str) -> List[Tuple[str, str]]:
    """Extract (src, en) sentence pairs from a TMX file."""
    pairs: List[Tuple[str, str]] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"[data] WARNING: TMX parse error {path}: {e}")
        return pairs

    for tu in tree.iterfind(".//tu"):
        src_seg, en_seg = None, None
        for tuv in tu.findall("tuv"):
            lang_attr = (
                tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                or tuv.get("lang") or ""
            ).lower()
            seg_el = tuv.find("seg")
            if seg_el is None:
                continue
            text = _unescape("".join(seg_el.itertext()))
            if not text:
                continue
            if lang_attr.startswith(src_lang):
                src_seg = text
            elif lang_attr.startswith("en"):
                en_seg = text
        if src_seg and en_seg:
            pairs.append((src_seg, en_seg))
    return pairs


def _read_tilde(root: str) -> List[DataRecord]:
    """Tilde TMX files: infer language from filename pattern <name>.<X>-en.tmx
    or <name>.en-<X>.tmx.  The non-English code is the source language."""
    d = os.path.join(root, "tilde_model_corpus")
    if not os.path.isdir(d):
        return []
    records: List[DataRecord] = []
    tmx_files = sorted(f for f in os.listdir(d) if f.endswith(".tmx"))
    for fname in tmx_files:
        name_body = fname[:-4]  # strip .tmx
        lang_code = None
        for part in name_body.split("."):
            if "-" not in part:
                continue
            a, b = part.split("-", 1)
            if b.lower() == "en" and a.lower() in LANG_PAIRS:
                lang_code = a.lower()
                break
            if a.lower() == "en" and b.lower() in LANG_PAIRS:
                lang_code = b.lower()
                break
        if lang_code is None:
            continue
        path = os.path.join(d, fname)
        pairs = _parse_tmx(path, lang_code)
        ds_label = f"tilde_{name_body}"
        for src, en in pairs:
            records.append((src, en, lang_code, ds_label))
        print(f"[data]   tilde  {lang_code}-en  {fname}: {len(pairs):>8,}")
    return records


# ==============================================================
# Fixed test-set persistence (sampled from corpus)
# ==============================================================

def save_test_split(test_records: List[DataRecord], test_dir: str):
    """Persist the test split as one TSV per language pair.

    Format: src <TAB> en <TAB> pair <TAB> dataset   (with header row).
    These files are the fixed test set — once written they are reused
    across all subsequent runs.
    """
    os.makedirs(test_dir, exist_ok=True)
    by_lang: Dict[str, List[DataRecord]] = defaultdict(list)
    for rec in test_records:
        by_lang[rec[2]].append(rec)

    for lang in LANG_PAIRS:
        recs = by_lang.get(lang, [])
        out_path = os.path.join(test_dir, f"{lang}-en.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("src\ten\tpair\tdataset\n")
            for src, en, lg, ds in recs:
                f.write(f"{src}\t{en}\t{lg}-en\t{ds}\n")
        print(f"[data]   Saved test {lang}-en : {len(recs):>7,} pairs → {out_path}")


def load_test_sets(test_dir: str):
    """Load the fixed test TSVs (previously saved by save_test_split).

    Returns (src_lines, tgt_lines, pair_labels, dataset_labels).
    pair_labels    e.g. 'de-en'           — for per-language evaluation.
    dataset_labels e.g. 'europarl_v10'    — identifies the source corpus.
    """
    src, tgt, pairs, datasets = [], [], [], []
    for lang in LANG_PAIRS:
        path = os.path.join(test_dir, f"{lang}-en.tsv")
        if not os.path.isfile(path):
            print(f"[data] WARNING: missing test set {path}")
            continue
        first = True
        with open(path, encoding="utf-8") as f:
            for line in f:
                if first:
                    first = False
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 4:
                    continue
                src.append(cols[0])
                tgt.append(cols[1])
                pairs.append(cols[2])
                datasets.append(cols[3])
    return src, tgt, pairs, datasets


def _test_sets_exist(test_dir: str) -> bool:
    """Return True if all 5 test TSV files already exist."""
    return all(
        os.path.isfile(os.path.join(test_dir, f"{lang}-en.tsv"))
        for lang in LANG_PAIRS
    )


# ==============================================================
# Stratified train / val split
# ==============================================================

def _stratified_split(records, val_ratio=0.01, test_ratio=0.01, seed=42):
    """Split records into train / val / test, stratified by (lang, dataset).

    For every unique (lang, dataset_name) group the rows are shuffled
    (deterministic via *seed*) and then partitioned:
        test  →  first  *test_ratio*  of each group
        val   →  next   *val_ratio*   of each group
        train →  remainder

    This ensures every corpus and every language pair are proportionally
    represented in all three splits.
    """
    rng = random.Random(seed)

    # Group indices by (lang, dataset)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, (_, _, lang, ds) in enumerate(records):
        groups[(lang, ds)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for key, indices in sorted(groups.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        n_val = max(1, int(len(shuffled) * val_ratio))
        test_idx.extend(shuffled[:n_test])
        val_idx.extend(shuffled[n_test:n_test + n_val])
        train_idx.extend(shuffled[n_test + n_val:])

    # Sort so order is deterministic
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    # Print summary grouped by language
    by_lang_train: Dict[str, int] = defaultdict(int)
    by_lang_val: Dict[str, int] = defaultdict(int)
    by_lang_test: Dict[str, int] = defaultdict(int)
    for i in train_idx:
        by_lang_train[records[i][2]] += 1
    for i in val_idx:
        by_lang_val[records[i][2]] += 1
    for i in test_idx:
        by_lang_test[records[i][2]] += 1
    for lang in LANG_PAIRS:
        tr_n = by_lang_train.get(lang, 0)
        va_n = by_lang_val.get(lang, 0)
        te_n = by_lang_test.get(lang, 0)
        print(f"[data]   split  {lang}-en :  "
              f"train={tr_n:>10,}  val={va_n:>7,}  test={te_n:>7,}")

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


# ==============================================================
# Encoding cache
# ==============================================================

def _records_hash(records: List[DataRecord]) -> str:
    """Quick hash of a records list (samples head+tail+count)."""
    h = hashlib.sha256()
    for rec in records[:5000]:
        h.update(rec[0].encode("utf-8", "replace"))
        h.update(rec[1].encode("utf-8", "replace"))
    for rec in records[-1000:]:
        h.update(rec[0].encode("utf-8", "replace"))
    h.update(str(len(records)).encode())
    return h.hexdigest()[:16]


def _cache_key(sp_model_path: str, content_hash: str, max_lines: int) -> str:
    h = hashlib.sha256()
    with open(sp_model_path, "rb") as f:
        h.update(f.read())
    h.update(content_hash.encode())
    h.update(str(max_lines).encode())
    return h.hexdigest()[:16]


# ── Multiprocessing worker for parallel encoding ─────────────

def _encode_chunk(args):
    """Worker: encode a chunk of (src, tgt) text pairs.

    Each worker loads its own SentencePiece model (they're lightweight)
    to avoid pickling issues and GIL contention.
    """
    model_path, texts_src, texts_tgt = args
    sp = spm.SentencePieceProcessor(model_file=model_path)
    src_ids = [sp.encode(t, out_type=int) for t in texts_src]
    tgt_ids = [sp.encode(t, out_type=int) for t in texts_tgt]
    return src_ids, tgt_ids


def _encode_records(
    records: List[DataRecord],
    tokenizer: SPTokenizer,
    sp_model_path: str,
    cache_dir: str,
    label: str,
    max_lines: int = 0,
    num_workers: int = 16,
):
    """Encode (src, tgt) from records via SentencePiece; cache as .pt.

    Uses multiprocessing to parallelise encoding across CPU cores.
    Returns (src_ids, tgt_ids) — both list[list[int]].
    """
    content_hash = _records_hash(records)
    key = _cache_key(sp_model_path, content_hash, max_lines)
    cache_path = os.path.join(cache_dir, f"{label}_{key}.pt")

    if os.path.isfile(cache_path):
        print(f"[data] Loading cached {label} → {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["src"], data["tgt"]

    subset = records[:max_lines] if max_lines > 0 else records
    n = len(subset)

    # Decide parallelism: use multiprocessing for large sets, serial for small
    effective_workers = min(num_workers, max(1, n // 50_000))

    if effective_workers <= 1 or n < 100_000:
        # Small set — serial is fine
        src_ids, tgt_ids = [], []
        for src, tgt, _, _ in tqdm(subset, desc=f"Encode {label}", leave=False):
            src_ids.append(tokenizer.encode(src))
            tgt_ids.append(tokenizer.encode(tgt))
    else:
        # Large set — split into chunks and encode in parallel
        chunk_size = (n + effective_workers - 1) // effective_workers
        chunks = []
        for i in range(0, n, chunk_size):
            batch = subset[i:i + chunk_size]
            texts_src = [r[0] for r in batch]
            texts_tgt = [r[1] for r in batch]
            chunks.append((sp_model_path, texts_src, texts_tgt))

        print(f"[data] Encoding {label} ({n:,} pairs) with "
              f"{effective_workers} workers, chunk_size ~{chunk_size:,}")

        src_ids, tgt_ids = [], []
        with mp.Pool(processes=effective_workers) as pool:
            for i, (s, t) in enumerate(
                pool.imap(_encode_chunk, chunks), 1
            ):
                src_ids.extend(s)
                tgt_ids.extend(t)
                print(f"[data]   chunk {i}/{len(chunks)} done "
                      f"({len(src_ids):,} / {n:,})")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"src": src_ids, "tgt": tgt_ids}, cache_path)
    print(f"[data] Cached {label} ({len(src_ids):,} pairs) → {cache_path}")
    return src_ids, tgt_ids


# ==============================================================
# Dump plain text for SentencePiece training
# ==============================================================

def _dump_text(records: List[DataRecord], src_path: str, tgt_path: str):
    """Write src + tgt lines for SentencePiece training (skip if exists).

    Uses 8 MB write buffers to reduce I/O syscall overhead.
    """
    if os.path.isfile(src_path) and os.path.isfile(tgt_path):
        return
    BUF = 8 * 1024 * 1024  # 8 MB buffer
    with open(src_path, "w", encoding="utf-8", buffering=BUF) as fs, \
         open(tgt_path, "w", encoding="utf-8", buffering=BUF) as ft:
        for src, tgt, _, _ in records:
            fs.write(src + "\n")
            ft.write(tgt + "\n")
    print(f"[data] Wrote SP training text ({len(records):,} pairs)")


# ==============================================================
# High-level entry point  (same return signature as original)
# ==============================================================

def prepare_data(
    data_dir: str,
    sp_dir: str,
    vocab_size: int = 32000,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
    num_threads: int = 16,
    max_train_lines: int = 0,
    *,
    test_dir: str = None,
    wikimatrix_min_score: float = 1.05,
):
    """Load multilingual data, prepare test / train / val splits, tokenise.

    Parameters
    ----------
    data_dir            Root of the Datasets folder.
    sp_dir              Directory for the SentencePiece model + cache.
    vocab_size          Shared BPE vocabulary size.
    val_ratio           Fraction of each (lang, dataset) held out for validation.
    test_ratio          Fraction of each (lang, dataset) held out for testing.
    seed                Random seed.
    num_threads         SentencePiece training threads.
    max_train_lines     Cap training pairs (0 = no cap; handy for smoke tests).
    test_dir            Where to store the fixed test TSVs (default: <data_dir>/test_sets).
    wikimatrix_min_score  Minimum cosine score for WikiMatrix pairs.

    Returns  (backward-compatible with original data.py)
    -------
    tokenizer                          SPTokenizer
    (train_src_ids,  train_tgt_ids)    list[list[int]]
    (val_src_ids,    val_tgt_ids)      list[list[int]]
    (test_src_lines, test_tgt_lines)   list[str]  — raw text for metrics
    (test_src_ids,   test_tgt_ids)     list[list[int]]
    """
    os.makedirs(sp_dir, exist_ok=True)
    if test_dir is None:
        test_dir = os.path.join(data_dir, "test_sets")
    cache_dir = os.path.join(sp_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── 1. Load all parallel corpora ──────────────────────────
    print("\n" + "=" * 64)
    print("  Step 1 : Load parallel corpora")
    print("=" * 64)
    all_records: List[DataRecord] = []
    all_records += _read_europarl(data_dir)
    all_records += _read_news_commentary(data_dir)
    all_records += _read_wikimatrix(data_dir, min_score=wikimatrix_min_score)
    all_records += _read_tilde(data_dir)

    if not all_records:
        raise RuntimeError("[data] No records loaded — check data_dir path.")
    print(f"\n[data] Total raw records : {len(all_records):,}")

    # ── 2. Three-way stratified split (or load cached test) ───
    print("\n" + "=" * 64)
    print("  Step 2 : Stratified train / val / test split")
    print("=" * 64)

    if _test_sets_exist(test_dir):
        # Test TSVs already saved from a previous run — reuse them.
        # Re-do the same deterministic split to get train/val correctly
        # (the same seed + data = same indices).
        print(f"[data] Fixed test sets found in {test_dir} — reusing.")
        train_records, val_records, _ = _stratified_split(
            all_records, val_ratio, test_ratio, seed,
        )
        test_src_lines, test_tgt_lines, _, _ = load_test_sets(test_dir)
    else:
        # First run: sample test from the corpus and save.
        train_records, val_records, test_records = _stratified_split(
            all_records, val_ratio, test_ratio, seed,
        )
        save_test_split(test_records, test_dir)
        test_src_lines = [r[0] for r in test_records]
        test_tgt_lines = [r[1] for r in test_records]

    if max_train_lines > 0 and len(train_records) > max_train_lines:
        rng = random.Random(seed)
        rng.shuffle(train_records)
        train_records = train_records[:max_train_lines]
        print(f"[data] ⚠  Training capped to {max_train_lines:,}")
    print(f"[data] train={len(train_records):,}  val={len(val_records):,}  "
          f"test={len(test_src_lines):,}")

    # ── 3. SentencePiece tokeniser ────────────────────────────
    print("\n" + "=" * 64)
    print("  Step 3 : SentencePiece tokeniser")
    print("=" * 64)
    model_prefix = os.path.join(sp_dir, "sp_bpe")
    model_path = model_prefix + ".model"
    if not os.path.isfile(model_path):
        sp_src = os.path.join(sp_dir, "sp_train_src.txt")
        sp_tgt = os.path.join(sp_dir, "sp_train_tgt.txt")
        _dump_text(train_records, sp_src, sp_tgt)
        _train_sentencepiece(
            [sp_src, sp_tgt], model_prefix,
            vocab_size=vocab_size, num_threads=num_threads,
        )
    else:
        print(f"[data] Loaded cached SP model → {model_path}")

    tokenizer = SPTokenizer(model_path)
    print(f"[data] Vocab size : {tokenizer.vocab_size:,}")

    # ── 4. Encode all splits ──────────────────────────────────
    print("\n" + "=" * 64)
    print("  Step 4 : Encode")
    print("=" * 64)
    # Use ~12.5% of cores (16 / 128) to leave headroom for OS + I/O
    enc_workers = min(16, max(1, mp.cpu_count() // 8))
    tr_src, tr_tgt = _encode_records(
        train_records, tokenizer, model_path, cache_dir, "train",
        max_train_lines, num_workers=enc_workers,
    )
    va_src, va_tgt = _encode_records(
        val_records, tokenizer, model_path, cache_dir, "val", 0,
        num_workers=enc_workers,
    )

    # Build lightweight records for test encoding
    test_recs = [(s, t, "test", "corpus") for s, t in zip(test_src_lines, test_tgt_lines)]
    te_src, te_tgt = _encode_records(
        test_recs, tokenizer, model_path, cache_dir, "test", 0,
        num_workers=enc_workers,
    )

    print(f"\n[data] ── Final ──  "
          f"train: {len(tr_src):,}  val: {len(va_src):,}  "
          f"test: {len(te_src):,}")

    return (
        tokenizer,
        (tr_src, tr_tgt),
        (va_src, va_tgt),
        (test_src_lines, test_tgt_lines),
        (te_src, te_tgt),
    )


# ==============================================================
# Standalone: print dataset statistics
# ==============================================================

if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description="Inspect multilingual datasets")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--test-dir", default=None)
    a = p.parse_args()

    td = a.test_dir or os.path.join(a.data_dir, "test_sets")

    print("── Loading corpora ───────────────────────────────────")
    recs = []
    recs += _read_europarl(a.data_dir)
    recs += _read_news_commentary(a.data_dir)
    recs += _read_wikimatrix(a.data_dir)
    recs += _read_tilde(a.data_dir)

    by_lang: Dict[str, int] = defaultdict(int)
    for _, _, lang, _ in recs:
        by_lang[lang] += 1

    print("\n── Per-pair counts ───────────────────────────────────")
    for lang in LANG_PAIRS:
        print(f"  {lang}-en : {by_lang.get(lang, 0):>10,}")
    print(f"  TOTAL  : {len(recs):>10,}")

    print("\n── 3-way stratified split ────────────────────────────")
    train_recs, val_recs, test_recs = _stratified_split(recs)

    print(f"\n  Train  : {len(train_recs):>10,}")
    print(f"  Val    : {len(val_recs):>10,}")
    print(f"  Test   : {len(test_recs):>10,}")

    # Save (or show cached) test TSVs
    if not _test_sets_exist(td):
        print("\n── Saving test sets ──────────────────────────────────")
        save_test_split(test_recs, td)
    else:
        print(f"\n── Test sets already saved in {td} ──")
        ts, tt, tp, td2 = load_test_sets(td)
        by_pair: Dict[str, int] = defaultdict(int)
        for pair in tp:
            by_pair[pair] += 1
        for pair, cnt in sorted(by_pair.items()):
            print(f"  {pair} : {cnt:,}")
        print(f"  Total  : {len(ts):,}")
