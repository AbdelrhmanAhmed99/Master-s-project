#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained multilingual Seq2Seq checkpoint.

Produces per-language and combined metrics (BLEU, ROUGE) along with
diagnostic plots.

Usage
-----
  python evaluate.py \\
      --data-dir /workspace/Datasets \\
      --checkpoint ./checkpoints/best.pt \\
      --output-dir ./testing_res

  # Quick greedy eval
  python evaluate.py \\
      --data-dir /workspace/Datasets \\
      --checkpoint ./checkpoints/best.pt \\
      --output-dir ./testing_res \\
      --batch-decode

Produces
--------
  testing_res/
    predictions.txt                — predicted vs reference translations
    metrics.txt / .json            — combined + per-language BLEU & ROUGE
    bleu_scores.png                — bar chart of overall BLEU
    bleu_per_language.png          — grouped bar chart of per-language BLEU
    rouge_scores.png               — bar chart of overall ROUGE P/R/F
    rouge_per_language.png         — grouped bar chart of per-language ROUGE-L F
    length_distribution.png        — histogram of predicted vs reference lengths
    bleu4_distribution.png         — per-sentence BLEU-4 histogram
"""

import argparse
import datetime
import json
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from data import (
    SPTokenizer,
    TranslationDataset,
    get_dataloader,
    prepare_data,
    load_test_sets,
    LANG_PAIRS,
    PAD_ID, SOS_ID, EOS_ID,
)
from model import Seq2Seq


# ==============================================================
# Argument parser
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate multilingual Seq2Seq NMT (many → English)",
    )

    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--output-dir", type=str, default="./testing_res")
    p.add_argument("--sp-dir", type=str, default=None,
                   help="Directory with SentencePiece model "
                        "(default: inferred from checkpoint)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit evaluation to first N test sentences (quick run)")
    p.add_argument("--batch-decode", action="store_true", default=False,
                   help="Use greedy batch decoding instead of beam search "
                        "(faster)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ==============================================================
# Metric helpers
# ==============================================================

def compute_bleu(pred_tokens: list[str], ref_tokens: list[str]):
    """Sentence-level BLEU 1-4 with smoothing."""
    sf = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = [0.0] * 4
        w[n - 1] = 1.0
        scores[f"BLEU-{n}"] = sentence_bleu(
            [ref_tokens], pred_tokens, weights=tuple(w),
            smoothing_function=sf,
        )
    return scores


def compute_rouge(pred_str: str, ref_str: str, scorer):
    results = scorer.score(ref_str, pred_str)
    out = {}
    for key in ("rouge1", "rouge2", "rougeL"):
        out[key] = {
            "precision": results[key].precision,
            "recall": results[key].recall,
            "fmeasure": results[key].fmeasure,
        }
    return out


# ==============================================================
# Decode helpers
# ==============================================================

def ids_to_text(ids: list[int], tokenizer: SPTokenizer) -> str:
    return tokenizer.decode_ids(ids)


def ids_to_tokens(ids: list[int], tokenizer: SPTokenizer) -> list[str]:
    return [tokenizer.id_to_piece(i) for i in ids
            if i not in (PAD_ID, SOS_ID, EOS_ID)]


# ==============================================================
# Metric aggregation
# ==============================================================

def _aggregate_metrics(indices, all_bleu, all_rouge, all_pred_lengths,
                       all_ref_lengths, all_bleu4):
    """Compute averaged metrics for a subset of indices."""
    n = len(indices)
    if n == 0:
        return None

    bleu_avg = defaultdict(float)
    rouge_avg = {"rouge1": defaultdict(float),
                 "rouge2": defaultdict(float),
                 "rougeL": defaultdict(float)}

    for i in indices:
        for k, v in all_bleu[i].items():
            bleu_avg[k] += v
        for rk in rouge_avg:
            for sk in ("precision", "recall", "fmeasure"):
                rouge_avg[rk][sk] += all_rouge[i][rk][sk]

    bleu_avg = {k: v / n for k, v in bleu_avg.items()}
    rouge_avg = {rk: {sk: sv / n for sk, sv in svs.items()}
                 for rk, svs in rouge_avg.items()}

    pred_lens = [all_pred_lengths[i] for i in indices]
    ref_lens = [all_ref_lengths[i] for i in indices]
    bleu4s = [all_bleu4[i] for i in indices]

    return {
        "n": n,
        "bleu": dict(bleu_avg),
        "rouge": {rk: dict(sv) for rk, sv in rouge_avg.items()},
        "avg_pred_len": float(np.mean(pred_lens)),
        "avg_ref_len": float(np.mean(ref_lens)),
        "avg_bleu4": float(np.mean(bleu4s)),
    }


# ==============================================================
# Plotting
# ==============================================================

def plot_bleu(bleu_avg: dict, output_dir: str):
    keys = sorted(bleu_avg.keys())
    vals = [bleu_avg[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals,
                  color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    ax.set_ylabel("Score")
    ax.set_title("BLEU Scores (sentence-level avg, all languages)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "bleu_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_bleu_per_language(per_lang_metrics: dict, output_dir: str):
    """Grouped bar chart of BLEU-1..4 per language pair."""
    langs = [l for l in LANG_PAIRS if f"{l}-en" in per_lang_metrics]
    if not langs:
        return
    bleu_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    x = np.arange(len(langs))
    w = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, bk in enumerate(bleu_keys):
        vals = [per_lang_metrics[f"{l}-en"]["bleu"][bk] for l in langs]
        bars = ax.bar(x + j * w, vals, w, label=bk, color=colors[j])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"{l}→en" for l in langs])
    ax.set_ylabel("Score")
    ax.set_title("BLEU per Language Pair")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "bleu_per_language.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_rouge(rouge_avg: dict, output_dir: str):
    metrics = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    prec = [rouge_avg[m]["precision"] for m in metrics]
    rec = [rouge_avg[m]["recall"] for m in metrics]
    fmeasure = [rouge_avg[m]["fmeasure"] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, prec, w, label="Precision", color="#4c72b0")
    ax.bar(x, rec, w, label="Recall", color="#55a868")
    ax.bar(x + w, fmeasure, w, label="F-measure", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    top = max(max(prec), max(rec), max(fmeasure))
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)
    ax.set_ylabel("Score")
    ax.set_title("ROUGE Scores (all languages)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "rouge_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_rouge_per_language(per_lang_metrics: dict, output_dir: str):
    """Bar chart of ROUGE-L F per language pair."""
    langs = [l for l in LANG_PAIRS if f"{l}-en" in per_lang_metrics]
    if not langs:
        return
    vals = [per_lang_metrics[f"{l}-en"]["rouge"]["rougeL"]["fmeasure"]
            for l in langs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"{l}→en" for l in langs], vals, color="#55a868")
    ax.set_ylabel("ROUGE-L F-measure")
    ax.set_title("ROUGE-L F per Language Pair")
    top = max(vals) if vals else 1
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "rouge_per_language.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_length_distribution(pred_lengths, ref_lengths, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    top = max(max(pred_lengths, default=1), max(ref_lengths, default=1))
    bins = max(20, min(50, top))
    ax.hist(ref_lengths, bins=bins, alpha=0.6,
            label="Reference", color="#4c72b0")
    ax.hist(pred_lengths, bins=bins, alpha=0.6,
            label="Predicted", color="#c44e52")
    ax.set_xlabel("Sentence length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Length Distribution: Predicted vs Reference")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "length_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_score_histogram(scores_list, label, output_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores_list, bins=30, color="#55a868",
            edgecolor="black", alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {label}")
    ax.axvline(np.mean(scores_list), color="red", linestyle="--",
               label=f"Mean = {np.mean(scores_list):.4f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


# ==============================================================
# Main
# ==============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")

    # ── Load checkpoint ───────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    saved_args = ckpt.get("args", {})
    print(f"[eval] Loaded checkpoint: {args.checkpoint}  "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── Resolve SP model directory ────────────────────────────
    sp_dir = args.sp_dir or saved_args.get("sp_dir") or \
             os.path.join(os.path.dirname(args.checkpoint), "sp")

    # ── Data ──────────────────────────────────────────────────
    tokenizer, _, _, (test_src_lines, test_tgt_lines), \
        (test_src_ids, test_tgt_ids) = prepare_data(
            args.data_dir, sp_dir,
            vocab_size=saved_args.get("vocab_size", 32000),
            seed=args.seed,
        )

    # Also load pair/dataset labels for per-language evaluation
    test_dir = os.path.join(args.data_dir, "test_sets")
    _, _, pair_labels, dataset_labels = load_test_sets(test_dir)

    if args.max_samples:
        test_src_lines = test_src_lines[: args.max_samples]
        test_tgt_lines = test_tgt_lines[: args.max_samples]
        test_src_ids = test_src_ids[: args.max_samples]
        test_tgt_ids = test_tgt_ids[: args.max_samples]
        pair_labels = pair_labels[: args.max_samples]
        dataset_labels = dataset_labels[: args.max_samples]

    # ── Model ─────────────────────────────────────────────────
    vs = ckpt.get("vocab_size", tokenizer.vocab_size)
    model = Seq2Seq(
        src_vocab_size=vs,
        tgt_vocab_size=vs,
        emb_dim=saved_args.get("emb_dim", 256),
        hidden_size=saved_args.get("hidden_size", 512),
        n_layers=saved_args.get("n_layers", 1),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] Model loaded — "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # ── Decode ────────────────────────────────────────────────
    n_test = len(test_src_ids)
    print(f"[eval] Decoding {n_test} test sentences …")

    pred_texts: list[str] = []
    pred_tokens: list[list[str]] = []
    ref_texts: list[str] = test_tgt_lines
    ref_tokens: list[list[str]] = []

    for tgt_ids in test_tgt_ids:
        ref_tokens.append(ids_to_tokens(tgt_ids, tokenizer))

    if args.batch_decode:
        test_ds = TranslationDataset(test_src_ids, test_tgt_ids)
        test_loader = get_dataloader(test_ds, args.batch_size,
                                     shuffle=False, num_workers=0)

        for src, src_lengths, _, _ in test_loader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            decoded = model.greedy_decode(
                src, src_lengths, max_len=args.max_len)
            for i in range(decoded.size(0)):
                ids = decoded[i].tolist()
                pred_texts.append(ids_to_text(ids, tokenizer))
                pred_tokens.append(ids_to_tokens(ids, tokenizer))
    else:
        for i, src_ids in enumerate(test_src_ids):
            results = model.beam_decode(
                src_ids, device,
                beam_size=args.beam_size,
                max_len=args.max_len)
            if results:
                best_ids = results[0][0]
                pred_texts.append(ids_to_text(best_ids, tokenizer))
                pred_tokens.append(ids_to_tokens(best_ids, tokenizer))
            else:
                pred_texts.append("")
                pred_tokens.append([])

            if (i + 1) % 500 == 0 or i == 0:
                print(f"  [{i+1}/{n_test}]  pred: "
                      f"{pred_texts[-1][:80]}…")

    # ── Write predictions ─────────────────────────────────────
    pred_path = os.path.join(args.output_dir, "predictions.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        for pt, rt, pl in zip(pred_texts, ref_texts, pair_labels):
            f.write(f"[{pl}]\n")
            f.write(f"PRED: {pt}\n")
            f.write(f"REF:  {rt}\n\n")
    print(f"[eval] Predictions → {pred_path}")

    # ── Compute per-sentence metrics ──────────────────────────
    print("[eval] Computing BLEU & ROUGE …")
    r_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False,
    )

    all_bleu: list[dict] = []
    all_rouge: list[dict] = []
    all_bleu4: list[float] = []
    all_pred_lengths: list[int] = []
    all_ref_lengths: list[int] = []

    for pred_tok, ref_tok, pred_str, ref_str in zip(
            pred_tokens, ref_tokens, pred_texts, ref_texts):
        all_pred_lengths.append(len(pred_tok))
        all_ref_lengths.append(len(ref_tok))

        if not pred_tok:
            b = {"BLEU-1": 0.0, "BLEU-2": 0.0,
                 "BLEU-3": 0.0, "BLEU-4": 0.0}
            all_bleu4.append(0.0)
        else:
            b = compute_bleu(pred_tok, ref_tok)
            all_bleu4.append(b["BLEU-4"])
        all_bleu.append(b)

        p_str = pred_str if pred_str else "<empty>"
        all_rouge.append(compute_rouge(p_str, ref_str, r_scorer))

    # ── Aggregate: combined ───────────────────────────────────
    all_indices = list(range(len(all_bleu)))
    combined = _aggregate_metrics(
        all_indices, all_bleu, all_rouge,
        all_pred_lengths, all_ref_lengths, all_bleu4,
    )

    # ── Aggregate: per language pair ──────────────────────────
    pair_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, pl in enumerate(pair_labels):
        pair_to_indices[pl].append(i)

    per_lang_metrics: dict[str, dict] = {}
    for pair, indices in sorted(pair_to_indices.items()):
        per_lang_metrics[pair] = _aggregate_metrics(
            indices, all_bleu, all_rouge,
            all_pred_lengths, all_ref_lengths, all_bleu4,
        )

    # ── Print & save metrics ──────────────────────────────────
    n = combined["n"]
    bleu_avg = combined["bleu"]
    rouge_avg = combined["rouge"]

    lines = [
        "=" * 72,
        "  Multilingual NMT Evaluation Report  (many → English)",
        "=" * 72,
        "",
        f"  Date           : "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Checkpoint     : {args.checkpoint}",
        f"  Total test     : {n:,}",
        f"  Decode method  : "
        f"{'greedy' if args.batch_decode else f'beam (k={args.beam_size})'}",
        f"  Tokenizer      : SentencePiece BPE "
        f"(vocab {tokenizer.vocab_size:,})",
        "",
        "-" * 72,
        "  COMBINED METRICS (all languages)",
        "-" * 72,
        "",
        f"  BLEU-1: {bleu_avg['BLEU-1']:.4f}",
        f"  BLEU-2: {bleu_avg['BLEU-2']:.4f}",
        f"  BLEU-3: {bleu_avg['BLEU-3']:.4f}",
        f"  BLEU-4: {bleu_avg['BLEU-4']:.4f}",
        "",
    ]
    for rk in ("rouge1", "rouge2", "rougeL"):
        label = rk.upper().replace("ROUGE", "ROUGE-")
        p = rouge_avg[rk]["precision"]
        r = rouge_avg[rk]["recall"]
        f_val = rouge_avg[rk]["fmeasure"]
        lines.append(f"  {label}  P:{p:.4f}  R:{r:.4f}  F:{f_val:.4f}")

    lines.extend([
        "",
        f"  Avg predicted length (subwords): "
        f"{combined['avg_pred_len']:.1f}",
        f"  Avg reference length (subwords): "
        f"{combined['avg_ref_len']:.1f}",
    ])

    # Per-language results
    lines.extend([
        "", "-" * 72, "  PER-LANGUAGE METRICS", "-" * 72, "",
    ])

    header = (f"  {'Pair':<8} {'N':>8}  "
              f"{'BLEU-1':>7} {'BLEU-2':>7} "
              f"{'BLEU-3':>7} {'BLEU-4':>7}  "
              f"{'R1-F':>6} {'R2-F':>6} {'RL-F':>6}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for pair in sorted(per_lang_metrics.keys()):
        m = per_lang_metrics[pair]
        b = m["bleu"]
        r = m["rouge"]
        lines.append(
            f"  {pair:<8} {m['n']:>8,}  "
            f"{b['BLEU-1']:>7.4f} {b['BLEU-2']:>7.4f} "
            f"{b['BLEU-3']:>7.4f} {b['BLEU-4']:>7.4f}  "
            f"{r['rouge1']['fmeasure']:>6.4f} "
            f"{r['rouge2']['fmeasure']:>6.4f} "
            f"{r['rougeL']['fmeasure']:>6.4f}"
        )

    lines.extend(["", "=" * 72])

    report_text = "\n".join(lines)
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"\n[eval] Metrics → {metrics_path}")

    # JSON output
    json_data = {
        "combined": combined,
        "per_language": per_lang_metrics,
        "config": {
            "checkpoint": args.checkpoint,
            "decode_method": ("greedy" if args.batch_decode
                              else f"beam_{args.beam_size}"),
            "vocab_size": tokenizer.vocab_size,
            "n_test": n,
        },
    }
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[eval] JSON → {json_path}")

    # ── Plots ─────────────────────────────────────────────────
    print("[eval] Generating plots …")
    plot_bleu(bleu_avg, args.output_dir)
    plot_bleu_per_language(per_lang_metrics, args.output_dir)
    plot_rouge(rouge_avg, args.output_dir)
    plot_rouge_per_language(per_lang_metrics, args.output_dir)
    plot_length_distribution(
        all_pred_lengths, all_ref_lengths, args.output_dir)
    plot_score_histogram(
        all_bleu4, "BLEU-4", args.output_dir, "bleu4_distribution.png")

    print(f"\n[eval] All results saved to {args.output_dir}/")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
