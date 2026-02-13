"""
data.py — SentencePiece BPE tokenizer + dataset loading.

Trains a **shared** BPE tokenizer on both source (EN) and target (DE)
training data using SentencePiece.  A shared subword vocabulary is the
standard approach for NMT between related language pairs (EN/DE share
many cognates and morphemes).

Advantages over word-level vocabulary:
  • Open vocabulary — no OOV / <unk> for rare or unseen words.
  • Learns subword units in an unsupervised, data-driven way.
  • Shared vocab means a single embedding matrix can be used.
  • Compact (32-35K tokens) yet covers the full training corpus.
"""

import hashlib
import os
import random

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ── Special token IDs (match SentencePiece defaults) ──────────
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2   # <s>  / bos
EOS_ID = 3   # </s> / eos


# ==============================================================
# SentencePiece wrapper
# ==============================================================

class SPTokenizer:
    """Thin wrapper around a SentencePiece model."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Text → list of subword IDs."""
        return self.sp.encode(text, out_type=int)

    def decode_ids(self, ids: list[int]) -> str:
        """List of subword IDs → detokenised text string."""
        # Filter out special IDs before decoding
        clean = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self.sp.decode(clean)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)


def train_sentencepiece(
    src_path: str,
    tgt_path: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    num_threads: int = 4,
    input_sentence_size: int = 4_000_000,
):
    """Train a shared SentencePiece BPE model on src + tgt data.

    The model and vocab files are written to:
        <model_prefix>.model
        <model_prefix>.vocab
    """
    print(f"[data] Training SentencePiece ({model_type}, vocab={vocab_size}) …")
    spm.SentencePieceTrainer.Train(
        input=f"{src_path},{tgt_path}",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        # Important for large corpora
        train_extremely_large_corpus=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        # Special token IDs — match our constants
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=SOS_ID,
        eos_id=EOS_ID,
        num_threads=num_threads,
        # Full character coverage for EN+DE
        character_coverage=1.0,
    )
    print(f"[data] SentencePiece model saved → {model_prefix}.model")


# ==============================================================
# Reading raw text files
# ==============================================================

def read_lines(path: str) -> list[str]:
    """Read a text file, one sentence per line → list of raw strings."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ==============================================================
# Dataset / DataLoader
# ==============================================================

class TranslationDataset(torch.utils.data.Dataset):
    """Parallel sentence pairs stored as pre-encoded integer-ID lists."""

    def __init__(self, src_ids: list[list[int]], tgt_ids: list[list[int]]):
        assert len(src_ids) == len(tgt_ids)
        self.src = src_ids
        self.tgt = tgt_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch, pad_id: int = PAD_ID):
    """Collate (src_ids, tgt_ids) pairs into padded tensors.

    Returns:
        src:         (batch, src_len)
        src_lengths: (batch,)
        tgt_in:      (batch, tgt_len)  — <s> + tokens
        tgt_out:     (batch, tgt_len)  — tokens + </s>
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
# Encoding cache helpers
# ==============================================================

def _cache_key(sp_model_path: str, data_path: str, max_lines: int) -> str:
    """Deterministic hash so the cache auto-invalidates when the SP
    model or the raw data file changes (or max_lines changes)."""
    h = hashlib.sha256()
    # Hash the SP model binary
    with open(sp_model_path, "rb") as f:
        h.update(f.read())
    # Hash the raw text file content
    with open(data_path, "rb") as f:
        h.update(f.read())
    h.update(str(max_lines).encode())
    return h.hexdigest()[:16]


def _encode_and_cache(
    tokenizer: "SPTokenizer",
    raw_path: str,
    cache_path: str,
    sp_model_path: str,
    max_lines: int,
    label: str,
) -> list[list[int]]:
    """Encode a text file through SentencePiece.  Results are cached as a
    `.pt` file keyed by (SP model + raw data + max_lines) so re-runs
    skip the expensive encoding step entirely.

    Cache layout inside `cache_dir`:
        <label>_<hash>.pt   — list[list[int]]
        <label>_<hash>.meta — plain-text: hash, line count, raw path
    """
    key = _cache_key(sp_model_path, raw_path, max_lines)
    pt_path = cache_path.replace(".pt", f"_{key}.pt")

    if os.path.isfile(pt_path):
        print(f"[data] Loading cached encoded {label} → {pt_path}")
        return torch.load(pt_path, weights_only=False)

    # Encode from scratch
    lines = read_lines(raw_path)
    if max_lines > 0:
        lines = lines[:max_lines]
    ids = [tokenizer.encode(s) for s in tqdm(lines, desc=f"Encode {label}", leave=False)]

    # Save cache
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(ids, pt_path)
    # Human-readable metadata
    meta_path = pt_path.replace(".pt", ".meta")
    with open(meta_path, "w") as f:
        f.write(f"hash={key}\nlines={len(ids)}\nsource={raw_path}\n")
    print(f"[data] Cached encoded {label} ({len(ids):,} sents) → {pt_path}")
    return ids


# ==============================================================
# High-level data preparation used by train.py / evaluate.py
# ==============================================================

def prepare_data(
    data_dir: str,
    sp_dir: str,
    vocab_size: int = 32000,
    val_ratio: float = 0.01,
    seed: int = 42,
    num_threads: int = 4,
    max_train_lines: int = 0,
):
    """Load raw data, train (or load cached) SentencePiece model, encode
    all splits, and return everything needed for training / evaluation.

    Encoded token IDs are **cached as .pt files** inside ``sp_dir/cache/``
    so that subsequent runs skip the expensive SentencePiece encoding
    (which takes ~3-4 min for 4.5 M sentences).  The cache auto-invalidates
    when the SP model, raw data, or ``max_train_lines`` changes.

    Expected files in `data_dir`:
        train.en.txt, train.de.txt
        newstest2014.en.txt, newstest2014.de.txt

    Returns:
        tokenizer: SPTokenizer
        (train_src_ids, train_tgt_ids)   — list[list[int]]
        (val_src_ids,   val_tgt_ids)
        (test_src_lines, test_tgt_lines) — raw strings (for BLEU/ROUGE)
        (test_src_ids,  test_tgt_ids)    — encoded
    """
    os.makedirs(sp_dir, exist_ok=True)
    model_prefix = os.path.join(sp_dir, "sp_bpe")
    model_path = model_prefix + ".model"
    cache_dir = os.path.join(sp_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    src_train_path = os.path.join(data_dir, "train.en.txt")
    tgt_train_path = os.path.join(data_dir, "train.de.txt")
    src_test_path = os.path.join(data_dir, "newstest2014.en.txt")
    tgt_test_path = os.path.join(data_dir, "newstest2014.de.txt")

    # ── Train or load SentencePiece ───────────────────────────
    if not os.path.isfile(model_path):
        train_sentencepiece(
            src_train_path, tgt_train_path,
            model_prefix, vocab_size=vocab_size,
            num_threads=num_threads,
        )
    else:
        print(f"[data] Loading cached SP model → {model_path}")

    tokenizer = SPTokenizer(model_path)
    print(f"[data] Vocab size: {tokenizer.vocab_size:,}")

    # ── Encode training data (cached) ─────────────────────────
    if max_train_lines > 0:
        print(f"[data] ⚠ Limiting training data to {max_train_lines:,} lines")

    train_src_ids = _encode_and_cache(
        tokenizer, src_train_path,
        os.path.join(cache_dir, "train_src.pt"),
        model_path, max_train_lines, "train-src",
    )
    train_tgt_ids = _encode_and_cache(
        tokenizer, tgt_train_path,
        os.path.join(cache_dir, "train_tgt.pt"),
        model_path, max_train_lines, "train-tgt",
    )
    assert len(train_src_ids) == len(train_tgt_ids), \
        f"Mismatch: {len(train_src_ids)} src vs {len(train_tgt_ids)} tgt"

    # ── Train / val split ─────────────────────────────────────
    n = len(train_src_ids)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))

    val_idx = set(indices[:n_val])
    val_src_ids = [train_src_ids[i] for i in sorted(val_idx)]
    val_tgt_ids = [train_tgt_ids[i] for i in sorted(val_idx)]
    tr_src_ids = [train_src_ids[i] for i in range(n) if i not in val_idx]
    tr_tgt_ids = [train_tgt_ids[i] for i in range(n) if i not in val_idx]

    # ── Encode test data (cached) ─────────────────────────────
    test_src_lines = read_lines(src_test_path)
    test_tgt_lines = read_lines(tgt_test_path)
    test_src_ids = _encode_and_cache(
        tokenizer, src_test_path,
        os.path.join(cache_dir, "test_src.pt"),
        model_path, 0, "test-src",
    )
    test_tgt_ids = _encode_and_cache(
        tokenizer, tgt_test_path,
        os.path.join(cache_dir, "test_tgt.pt"),
        model_path, 0, "test-tgt",
    )

    print(f"[data] train: {len(tr_src_ids):,}  |  val: {len(val_src_ids):,}  "
          f"|  test: {len(test_src_ids):,}")

    return (
        tokenizer,
        (tr_src_ids, tr_tgt_ids),
        (val_src_ids, val_tgt_ids),
        (test_src_lines, test_tgt_lines),
        (test_src_ids, test_tgt_ids),
    )
