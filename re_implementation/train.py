#!/usr/bin/env python3
"""
train.py — Command-line training script for the Seq2Seq model.

Training can be controlled via **steps** or **epochs**:

  • --max-steps N     → train for exactly N optimizer steps then stop.
  • --epochs N        → converted to steps internally:
                        max_steps = epochs × ceil(len(train) / batch_size)

Gradient accumulation lets you simulate a large effective batch size on
limited GPU memory:

  effective_batch_size = batch_size × accum_steps

Usage examples
--------------
  # Epoch-based (default), effective batch = 64 × 4 = 256
  python train.py --data-dir ../data --save-dir ./checkpoints \\
      --batch-size 64 --accum-steps 4 --epochs 10

  # Step-based, train for 100 000 optimizer steps
  python train.py --data-dir ../data --save-dir ./checkpoints \\
      --batch-size 32 --accum-steps 8 --max-steps 100000

  # Resume from a checkpoint
  python train.py --data-dir ../data --save-dir ./checkpoints \\
      --resume ./checkpoints/best.pt
"""

import argparse
import datetime
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import (
    TranslationDataset,
    get_dataloader,
    prepare_data,
    PAD_ID,
)
from model import Seq2Seq


# ==============================================================
# Argument parser
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Seq2Seq EN→DE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory with train.en.txt, train.de.txt, newstest2014.*.txt")
    p.add_argument("--save-dir", type=str, default="./checkpoints",
                   help="Directory for checkpoints")
    p.add_argument("--sp-dir", type=str, default=None,
                   help="SentencePiece model dir (default: <save-dir>/sp)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path to resume from")

    # ── Tokenizer ─────────────────────────────────────────────
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="SentencePiece BPE vocabulary size")

    # ── Model ─────────────────────────────────────────────────
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)

    # ── Training schedule ─────────────────────────────────────
    g = p.add_mutually_exclusive_group()
    g.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs (converted to steps internally)")
    g.add_argument("--max-steps", type=int, default=None,
                   help="Total optimizer steps to train")

    # ── Batch / accumulation ──────────────────────────────────
    p.add_argument("--batch-size", type=int, default=64,
                   help="Micro-batch size per forward pass (controls DataLoader)")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps "
                        "(effective_batch = batch_size × accum_steps)")

    # ── Optimiser ─────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=5.0,
                   help="Max gradient norm for clipping")

    # ── Early stopping / checkpointing ────────────────────────
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (validation rounds)")
    p.add_argument("--val-ratio", type=float, default=0.01,
                   help="Fraction of training data held out for validation")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Run validation & checkpoint every N steps "
                        "(default: once per epoch)")
    p.add_argument("--save-every", type=int, default=None,
                   help="Save a numbered checkpoint every N steps "
                        "(in addition to best/last)")

    # ── Logging ───────────────────────────────────────────────
    p.add_argument("--log-every", type=int, default=100,
                   help="Print training stats every N steps")

    # ── Misc ──────────────────────────────────────────────────
    p.add_argument("--amp", action="store_true", default=False,
                   help="Mixed-precision training (requires CUDA)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-lines", type=int, default=0,
                   help="Truncate training data to N lines (0 = use all). "
                        "Useful for quick smoke tests.")

    args = p.parse_args()

    # Default to 15 epochs if neither --epochs nor --max-steps is given
    if args.epochs is None and args.max_steps is None:
        args.epochs = 15

    return args


# ==============================================================
# Helpers
# ==============================================================

def compute_loss(logits, targets):
    """Cross-entropy loss ignoring PAD positions.
    logits:  (B, T, V)   targets: (B, T)
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
    )


def fmt_time(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))


def fmt_num(n) -> str:
    """Thousands-separated integer."""
    return f"{int(n):,}"


def save_checkpoint(path, model, optimizer, scaler, global_step, epoch,
                    best_val, vocab_size, args):
    torch.save({
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "vocab_size": vocab_size,
        "args": vars(args),
    }, path)


# ==============================================================
# Validation
# ==============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    n_sents = 0

    for src, src_lengths, tgt_in, tgt_out in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(src, src_lengths, tgt_in)
            loss = compute_loss(logits, tgt_out)

        mask = (tgt_out != PAD_ID)
        n_tok = mask.sum().item()
        total_loss += loss.item() * n_tok
        n_tokens += n_tok
        n_sents += src.size(0)

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    return avg_loss, ppl, n_tokens, n_sents


# ==============================================================
# Training loop  (step-based with gradient accumulation)
# ==============================================================

def train(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    sp_dir = args.sp_dir or os.path.join(args.save_dir, "sp")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────
    print("=" * 64)
    print("  Seq2Seq Training")
    print("=" * 64)
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory      : {mem:.1f} GB")
    print(f"  Mixed precision : {'ON' if use_amp else 'OFF'}")
    if args.amp and not use_amp:
        print("  ⚠  --amp ignored (no CUDA device)")
    print()

    # ── Data ──────────────────────────────────────────────────
    tokenizer, (tr_s, tr_t), (va_s, va_t), _, _ = prepare_data(
        args.data_dir, sp_dir,
        vocab_size=args.vocab_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_lines=args.max_train_lines,
    )

    train_ds = TranslationDataset(tr_s, tr_t)
    val_ds = TranslationDataset(va_s, va_t)

    train_loader = get_dataloader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    val_loader = get_dataloader(val_ds, args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / args.accum_steps)

    # Resolve max_steps
    if args.max_steps is not None:
        max_steps = args.max_steps
        est_epochs = max_steps / steps_per_epoch
    else:
        max_steps = args.epochs * steps_per_epoch
        est_epochs = args.epochs

    effective_batch = args.batch_size * args.accum_steps

    # Resolve eval_every
    eval_every = args.eval_every or steps_per_epoch

    # ── Print data / schedule summary ─────────────────────────
    print("─" * 64)
    print("  Data")
    print("─" * 64)
    print(f"  Train sentences : {fmt_num(len(train_ds))}")
    print(f"  Val sentences   : {fmt_num(len(val_ds))}")
    print(f"  Vocab size      : {fmt_num(tokenizer.vocab_size)}")
    print()
    print("─" * 64)
    print("  Training schedule")
    print("─" * 64)
    print(f"  Micro-batch size      : {args.batch_size}")
    print(f"  Gradient accum steps  : {args.accum_steps}")
    print(f"  Effective batch size  : {effective_batch}")
    print(f"  Batches / epoch       : {fmt_num(batches_per_epoch)}")
    print(f"  Optimizer steps/epoch : {fmt_num(steps_per_epoch)}")
    print(f"  Max optimizer steps   : {fmt_num(max_steps)}")
    print(f"  ≈ Epochs              : {est_epochs:.1f}")
    print(f"  Eval / ckpt every     : {fmt_num(eval_every)} steps")
    if args.save_every:
        print(f"  Numbered ckpt every   : {fmt_num(args.save_every)} steps")
    print(f"  Log every             : {fmt_num(args.log_every)} steps")
    print(f"  Early stopping        : patience {args.patience}")
    print(f"  Learning rate         : {args.lr}")
    print(f"  Gradient clip norm    : {args.clip}")
    print()

    # ── Model ─────────────────────────────────────────────────
    vs = tokenizer.vocab_size
    model = Seq2Seq(
        src_vocab_size=vs,
        tgt_vocab_size=vs,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("─" * 64)
    print("  Model")
    print("─" * 64)
    print(f"  Architecture    : Seq2Seq + Bahdanau attention")
    print(f"  Embedding dim   : {args.emb_dim}")
    print(f"  Hidden size     : {args.hidden_size}")
    print(f"  Encoder layers  : {args.n_layers} (bidirectional)")
    print(f"  Dropout         : {args.dropout}")
    print(f"  Total params    : {fmt_num(total_params)}")
    print(f"  Trainable params: {fmt_num(trainable_params)}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────
    global_step = 0
    epoch = 0
    best_val = float("inf")
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[train] ✓ Resumed from {args.resume}")
        print(f"         step {fmt_num(global_step)}, epoch {epoch}, "
              f"best_val {best_val:.4f}")
        print()

    # ── Training ──────────────────────────────────────────────
    print("=" * 64)
    print(f"  Starting training  ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    print("=" * 64)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Running stats (reset every log window)
    run_loss = 0.0
    run_tokens = 0
    run_sents = 0
    micro_step = 0       # micro-batches since last optimizer step
    t_start = time.time()
    t_log = time.time()

    train_iter = iter(train_loader)

    while global_step < max_steps:
        # ── Fetch next micro-batch (cycle through epochs) ─────
        try:
            src, src_lengths, tgt_in, tgt_out = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            src, src_lengths, tgt_in, tgt_out = next(train_iter)

        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        # ── Forward / backward (accumulate) ───────────────────
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(src, src_lengths, tgt_in)
            loss = compute_loss(logits, tgt_out)
            # Normalise loss by accum_steps so gradients are averaged
            loss_scaled = loss / args.accum_steps

        scaler.scale(loss_scaled).backward()

        # Token / sentence bookkeeping (use un-scaled loss)
        mask = (tgt_out != PAD_ID)
        n_tok = mask.sum().item()
        run_loss += loss.item() * n_tok
        run_tokens += n_tok
        run_sents += src.size(0)
        micro_step += 1

        # ── Optimizer step (every accum_steps micro-batches) ──
        if micro_step % args.accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # ── Logging ───────────────────────────────────────
            if global_step % args.log_every == 0:
                elapsed = time.time() - t_log
                avg_loss = run_loss / max(run_tokens, 1)
                ppl = math.exp(min(avg_loss, 100))
                tok_s = run_tokens / max(elapsed, 1e-6)
                total_elapsed = time.time() - t_start
                est_remain = (total_elapsed / max(global_step, 1)) * (max_steps - global_step)
                cur_epoch = epoch + 1  # 1-indexed display

                pct = 100.0 * global_step / max_steps
                lr_now = optimizer.param_groups[0]["lr"]

                print(
                    f"  step {global_step:>7,}/{max_steps:,} ({pct:5.1f}%) │ "
                    f"epoch {cur_epoch:>3} │ "
                    f"loss {avg_loss:.4f} │ ppl {ppl:8.2f} │ "
                    f"gnorm {grad_norm:.2f} │ "
                    f"lr {lr_now:.2e} │ "
                    f"{tok_s:,.0f} tok/s │ "
                    f"{run_sents / max(elapsed, 1e-6):,.0f} sent/s │ "
                    f"elapsed {fmt_time(total_elapsed)} │ "
                    f"ETA {fmt_time(est_remain)}"
                )

                # Reset running window
                run_loss = 0.0
                run_tokens = 0
                run_sents = 0
                t_log = time.time()

            # ── Numbered checkpoint ───────────────────────────
            if args.save_every and global_step % args.save_every == 0:
                p = os.path.join(args.save_dir, f"step-{global_step}.pt")
                save_checkpoint(p, model, optimizer, scaler, global_step,
                                epoch + 1, best_val, vs, args)
                print(f"  💾  Checkpoint → {p}")

            # ── Validation & best-model checkpoint ────────────
            if global_step % eval_every == 0 or global_step >= max_steps:
                cur_epoch = epoch + 1
                print()
                print(f"  ── Validation at step {global_step:,} "
                      f"(epoch ≈ {cur_epoch}) ──")
                val_loss, val_ppl, val_tok, val_sents = validate(
                    model, val_loader, device, use_amp,
                )
                print(f"     val loss {val_loss:.4f} │ "
                      f"val ppl {val_ppl:.2f} │ "
                      f"tokens {fmt_num(val_tok)} │ "
                      f"sents {fmt_num(val_sents)}")

                # Save last
                last_path = os.path.join(args.save_dir, "last.pt")
                save_checkpoint(last_path, model, optimizer, scaler,
                                global_step, cur_epoch, best_val, vs, args)

                if val_loss < best_val:
                    delta = best_val - val_loss
                    best_val = val_loss
                    best_path = os.path.join(args.save_dir, "best.pt")
                    save_checkpoint(best_path, model, optimizer, scaler,
                                    global_step, cur_epoch, best_val, vs, args)
                    print(f"     ✓ New best (Δ{delta:.4f}) → {best_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"     ✗ No improvement "
                          f"(patience {patience_counter}/{args.patience})")

                if patience_counter >= args.patience:
                    print()
                    print(f"  ⏹  Early stopping — no improvement for "
                          f"{args.patience} evaluations.")
                    break

                model.train()
                print()

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - t_start
    print("=" * 64)
    print("  Training complete")
    print("=" * 64)
    print(f"  Total steps     : {fmt_num(global_step)}")
    print(f"  Total epochs    : ≈ {epoch + 1}")
    print(f"  Best val loss   : {best_val:.4f}")
    print(f"  Best val ppl    : {math.exp(min(best_val, 100)):.2f}")
    print(f"  Wall time       : {fmt_time(total_time)}")
    print(f"  Checkpoints in  : {args.save_dir}/")
    print()


# ==============================================================
# Main
# ==============================================================

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
