#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained Seq2Seq checkpoint and save metric plots.

Usage
-----
  python evaluate.py \\
      --data-dir ../data \\
      --checkpoint ./checkpoints/best.pt \\
      --output-dir ./testing_res

Produces
--------
  testing_res/
    predictions.txt          — predicted vs reference translations
    metrics.txt / .json      — BLEU-1/2/3/4, ROUGE-1/2/L numbers
    bleu_scores.png          — bar chart of BLEU scores
    rouge_scores.png         — bar chart of ROUGE P/R/F
    length_distribution.png  — histogram of predicted vs reference lengths
    bleu4_distribution.png   — per-sentence BLEU-4 histogram
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
    PAD_ID, SOS_ID, EOS_ID,
)
from model import Seq2Seq


# ==============================================================
# Argument parser
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Seq2Seq EN→DE")

    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--output-dir", type=str, default="./testing_res")
    p.add_argument("--sp-dir", type=str, default=None,
                   help="Directory with SentencePiece model (default: inferred from checkpoint)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit evaluation to first N test sentences (quick run)")
    p.add_argument("--batch-decode", action="store_true", default=False,
                   help="Use greedy batch decoding instead of beam search (faster)")
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
    """Convert subword IDs → readable text using SentencePiece detokeniser."""
    return tokenizer.decode_ids(ids)


def ids_to_tokens(ids: list[int], tokenizer: SPTokenizer) -> list[str]:
    """Convert subword IDs → list of subword pieces (for BLEU computation)."""
    return [tokenizer.id_to_piece(i) for i in ids
            if i not in (PAD_ID, SOS_ID, EOS_ID)]


# ==============================================================
# Plotting
# ==============================================================

def plot_bleu(bleu_avg: dict, output_dir: str):
    keys = sorted(bleu_avg.keys())
    vals = [bleu_avg[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    ax.set_ylabel("Score")
    ax.set_title("BLEU Scores (sentence-level avg)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "bleu_scores.png")
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
    ax.set_title("ROUGE Scores")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "rouge_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_length_distribution(pred_lengths, ref_lengths, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    top = max(max(pred_lengths, default=1), max(ref_lengths, default=1))
    bins = max(20, min(50, top))
    ax.hist(ref_lengths, bins=bins, alpha=0.6, label="Reference", color="#4c72b0")
    ax.hist(pred_lengths, bins=bins, alpha=0.6, label="Predicted", color="#c44e52")
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
    ax.hist(scores_list, bins=30, color="#55a868", edgecolor="black", alpha=0.7)
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


def plot_comparison(bleu_avg, rouge_avg, paper_metrics, rouge_key_map, output_dir):
    """Side-by-side bar chart: ours vs paper for BLEU and ROUGE-F."""
    labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
              "ROUGE-1 F", "ROUGE-2 F", "ROUGE-L F"]
    ours_vals = [
        bleu_avg["BLEU-1"], bleu_avg["BLEU-2"],
        bleu_avg["BLEU-3"], bleu_avg["BLEU-4"],
        rouge_avg["rouge1"]["fmeasure"],
        rouge_avg["rouge2"]["fmeasure"],
        rouge_avg["rougeL"]["fmeasure"],
    ]
    paper_vals = [
        paper_metrics["BLEU-1"], paper_metrics["BLEU-2"],
        paper_metrics["BLEU-3"], paper_metrics["BLEU-4"],
        paper_metrics["ROUGE-1"]["fmeasure"],
        paper_metrics["ROUGE-2"]["fmeasure"],
        paper_metrics["ROUGE-L"]["fmeasure"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, ours_vals, w, label="Ours", color="#4c72b0")
    bars2 = ax.bar(x + w/2, paper_vals, w, label="Paper (Mini-Former)", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Our Results vs Paper (Mini-Former)")
    ax.legend()
    top = max(max(ours_vals, default=0), max(paper_vals, default=0))
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)
    for bar, v in zip(bars1, ours_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars2, paper_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_chart.png")
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    print(f"[eval] Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # ── Resolve SP model directory ────────────────────────────
    sp_dir = args.sp_dir or saved_args.get("sp_dir") or \
             os.path.join(os.path.dirname(args.checkpoint), "sp")

    # ── Data ──────────────────────────────────────────────────
    tokenizer, _, _, (test_src_lines, test_tgt_lines), (test_src_ids, test_tgt_ids) = \
        prepare_data(
            args.data_dir, sp_dir,
            vocab_size=saved_args.get("vocab_size", 32000),
            seed=args.seed,
        )

    if args.max_samples:
        test_src_lines = test_src_lines[: args.max_samples]
        test_tgt_lines = test_tgt_lines[: args.max_samples]
        test_src_ids = test_src_ids[: args.max_samples]
        test_tgt_ids = test_tgt_ids[: args.max_samples]

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
    print(f"[eval] Model loaded — {sum(p.numel() for p in model.parameters()):,} params")

    # ── Decode ────────────────────────────────────────────────
    n_test = len(test_src_ids)
    print(f"[eval] Decoding {n_test} test sentences …")

    pred_texts: list[str] = []   # detokenised strings
    pred_tokens: list[list[str]] = []
    ref_texts: list[str] = test_tgt_lines  # original reference strings
    ref_tokens: list[list[str]] = []

    # Tokenise references the same way (subword pieces) for fair BLEU
    for tgt_ids in test_tgt_ids:
        ref_tokens.append(ids_to_tokens(tgt_ids, tokenizer))

    if args.batch_decode:
        # Greedy batch decoding
        test_ds = TranslationDataset(test_src_ids, test_tgt_ids)
        test_loader = get_dataloader(test_ds, args.batch_size,
                                     shuffle=False, num_workers=0)

        for src, src_lengths, _, _ in test_loader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            decoded = model.greedy_decode(src, src_lengths, max_len=args.max_len)
            for i in range(decoded.size(0)):
                ids = decoded[i].tolist()
                pred_texts.append(ids_to_text(ids, tokenizer))
                pred_tokens.append(ids_to_tokens(ids, tokenizer))
    else:
        # Beam search — sentence by sentence
        for i, src_ids in enumerate(test_src_ids):
            results = model.beam_decode(src_ids, device,
                                        beam_size=args.beam_size,
                                        max_len=args.max_len)
            if results:
                best_ids = results[0][0]
                pred_texts.append(ids_to_text(best_ids, tokenizer))
                pred_tokens.append(ids_to_tokens(best_ids, tokenizer))
            else:
                pred_texts.append("")
                pred_tokens.append([])

            if (i + 1) % 200 == 0 or i == 0:
                print(f"  [{i+1}/{n_test}]  pred: {pred_texts[-1][:80]}…")

    # ── Write predictions ─────────────────────────────────────
    pred_path = os.path.join(args.output_dir, "predictions.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        for pt, rt in zip(pred_texts, ref_texts):
            f.write(f"PRED: {pt}\n")
            f.write(f"REF:  {rt}\n\n")
    print(f"[eval] Predictions → {pred_path}")

    # ── Compute metrics ───────────────────────────────────────
    print("[eval] Computing BLEU & ROUGE …")
    r_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False,
    )

    bleu_accum = defaultdict(float)
    rouge_accum = {"rouge1": defaultdict(float),
                   "rouge2": defaultdict(float),
                   "rougeL": defaultdict(float)}
    bleu4_per_sent: list[float] = []
    pred_lengths: list[int] = []
    ref_lengths: list[int] = []
    n = len(pred_tokens)

    for pred_tok, ref_tok, pred_str, ref_str in zip(
            pred_tokens, ref_tokens, pred_texts, ref_texts):
        pred_lengths.append(len(pred_tok))
        ref_lengths.append(len(ref_tok))

        # BLEU (on subword pieces — standard practice)
        if not pred_tok:
            for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"):
                bleu_accum[k] += 0.0
            bleu4_per_sent.append(0.0)
        else:
            b = compute_bleu(pred_tok, ref_tok)
            for k, v in b.items():
                bleu_accum[k] += v
            bleu4_per_sent.append(b["BLEU-4"])

        # ROUGE (on detokenised text — measures surface overlap)
        p_str = pred_str if pred_str else "<empty>"
        r = compute_rouge(p_str, ref_str, r_scorer)
        for rk in rouge_accum:
            for sk in ("precision", "recall", "fmeasure"):
                rouge_accum[rk][sk] += r[rk][sk]

    bleu_avg = {k: v / n for k, v in bleu_accum.items()}
    rouge_avg = {rk: {sk: sv / n for sk, sv in svs.items()}
                 for rk, svs in rouge_accum.items()}

    # ── Print & save metrics ──────────────────────────────────
    lines = [
        f"Evaluated {n} sentence pairs",
        f"Tokenizer: SentencePiece BPE (vocab {tokenizer.vocab_size:,})",
        "",
        f"BLEU-1: {bleu_avg['BLEU-1']:.4f}",
        f"BLEU-2: {bleu_avg['BLEU-2']:.4f}",
        f"BLEU-3: {bleu_avg['BLEU-3']:.4f}",
        f"BLEU-4: {bleu_avg['BLEU-4']:.4f}",
        "",
    ]
    for rk in ("rouge1", "rouge2", "rougeL"):
        label = rk.upper().replace("ROUGE", "ROUGE-")
        p = rouge_avg[rk]["precision"]
        r = rouge_avg[rk]["recall"]
        f = rouge_avg[rk]["fmeasure"]
        lines.append(f"{label}  P:{p:.4f}  R:{r:.4f}  F:{f:.4f}")

    lines.append("")
    lines.append(f"Avg predicted length (subwords): {np.mean(pred_lengths):.1f}")
    lines.append(f"Avg reference length (subwords): {np.mean(ref_lengths):.1f}")

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[eval] Metrics → {metrics_path}")

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump({"bleu": bleu_avg, "rouge": rouge_avg, "n": n}, f, indent=2)

    # ── Paper comparison report ───────────────────────────────
    paper_metrics = {
        "BLEU-1": 0.42, "BLEU-2": 0.22, "BLEU-3": 0.14, "BLEU-4": 0.09,
        "ROUGE-1": {"precision": 0.59, "recall": 0.44, "fmeasure": 0.49},
        "ROUGE-2": {"precision": 0.28, "recall": 0.23, "fmeasure": 0.25},
        "ROUGE-L": {"precision": 0.57, "recall": 0.42, "fmeasure": 0.46},
    }

    rouge_key_map = {"ROUGE-1": "rouge1", "ROUGE-2": "rouge2", "ROUGE-L": "rougeL"}

    report_lines = [
        "=" * 72,
        "  EXPERIMENT REPORT — Seq2Seq EN→DE  (BPE re-implementation)",
        "=" * 72, "",
        f"  Date          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Checkpoint    : {args.checkpoint}",
        f"  Test sentences: {n}",
        f"  Beam size     : {args.beam_size if not args.batch_decode else 'greedy'}",
        f"  Tokenizer     : SentencePiece BPE (vocab {tokenizer.vocab_size:,})",
        "", "-" * 72,
        f"  {'Metric':<12} {'Ours':>10} {'Paper':>10} {'Δ':>10} {'% of Paper':>12}",
        "-" * 72,
    ]

    for bk in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"):
        ours = bleu_avg[bk]
        paper = paper_metrics[bk]
        delta = ours - paper
        pct = (ours / paper * 100) if paper > 0 else 0
        report_lines.append(f"  {bk:<12} {ours:>10.4f} {paper:>10.4f} {delta:>+10.4f} {pct:>11.1f}%")

    report_lines.append("")
    for rk_label, rk_key in rouge_key_map.items():
        for sk, sk_label in [("precision", "P"), ("recall", "R"), ("fmeasure", "F")]:
            ours = rouge_avg[rk_key][sk]
            paper = paper_metrics[rk_label][sk]
            delta = ours - paper
            pct = (ours / paper * 100) if paper > 0 else 0
            tag = f"{rk_label}-{sk_label}"
            report_lines.append(f"  {tag:<12} {ours:>10.4f} {paper:>10.4f} {delta:>+10.4f} {pct:>11.1f}%")
        report_lines.append("")

    # Overall closeness summary
    bleu_pcts = []
    for bk in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"):
        bleu_pcts.append(bleu_avg[bk] / paper_metrics[bk] * 100 if paper_metrics[bk] > 0 else 0)
    rouge_f_pcts = []
    for rk_label, rk_key in rouge_key_map.items():
        ours_f = rouge_avg[rk_key]["fmeasure"]
        paper_f = paper_metrics[rk_label]["fmeasure"]
        rouge_f_pcts.append(ours_f / paper_f * 100 if paper_f > 0 else 0)

    avg_bleu_pct = sum(bleu_pcts) / len(bleu_pcts)
    avg_rouge_pct = sum(rouge_f_pcts) / len(rouge_f_pcts)

    report_lines.extend([
        "-" * 72,
        f"  Average BLEU   closeness to paper : {avg_bleu_pct:.1f}%",
        f"  Average ROUGE-F closeness to paper : {avg_rouge_pct:.1f}%",
        f"  Overall closeness                  : {(avg_bleu_pct + avg_rouge_pct) / 2:.1f}%",
        "-" * 72, "",
    ])

    if avg_bleu_pct >= 90 and avg_rouge_pct >= 90:
        report_lines.append("  ✓ Results are VERY CLOSE to the paper (≥90%)")
    elif avg_bleu_pct >= 70 and avg_rouge_pct >= 70:
        report_lines.append("  ~ Results are REASONABLY CLOSE to the paper (≥70%)")
    elif avg_bleu_pct >= 50 and avg_rouge_pct >= 50:
        report_lines.append("  △ Results are PARTIALLY matching the paper (≥50%)")
    else:
        report_lines.append("  ✗ Results are BELOW the paper's reported numbers (<50%)")

    report_lines.append("")
    report_lines.append("=" * 72)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"\n[eval] Comparison report → {report_path}")

    # Also save paper metrics alongside ours in JSON
    comparison_json = {
        "ours": {"bleu": bleu_avg, "rouge": rouge_avg, "n": n},
        "paper": paper_metrics,
        "closeness": {
            "avg_bleu_pct": round(avg_bleu_pct, 2),
            "avg_rouge_f_pct": round(avg_rouge_pct, 2),
            "overall_pct": round((avg_bleu_pct + avg_rouge_pct) / 2, 2),
        },
    }
    cmp_json_path = os.path.join(args.output_dir, "comparison.json")
    with open(cmp_json_path, "w") as f:
        json.dump(comparison_json, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    print("[eval] Generating plots …")
    plot_bleu(bleu_avg, args.output_dir)
    plot_rouge(rouge_avg, args.output_dir)
    plot_length_distribution(pred_lengths, ref_lengths, args.output_dir)
    plot_score_histogram(bleu4_per_sent, "BLEU-4", args.output_dir,
                         "bleu4_distribution.png")
    plot_comparison(bleu_avg, rouge_avg, paper_metrics, rouge_key_map, args.output_dir)

    print(f"\n[eval] All results saved to {args.output_dir}/")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
