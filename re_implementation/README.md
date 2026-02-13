# Seq2Seq Machine Translation — Re-implementation

Command-line **EN → DE** translation using a Seq2Seq + Attention model (PyTorch).

Uses a **shared SentencePiece BPE tokenizer** trained unsupervised on both
source and target data — eliminates OOV issues and produces a compact subword
vocabulary.

---

## Quick Reference

| Task | Command |
|------|---------|
| Run full experiment (train + eval) | `screen -S experiment bash run_experiment.sh` |
| Evaluate latest checkpoint | `python evaluate.py --data-dir ../data --checkpoint ./checkpoints/best.pt --output-dir ./testing_res --beam-size 5` |
| Quick greedy eval (fast) | `python evaluate.py --data-dir ../data --checkpoint ./checkpoints/best.pt --output-dir ./testing_res --batch-decode` |
| Check training progress | `tail -20 experiment_full.log` |
| Live training log | `tail -f experiment_full.log` |
| Reattach to screen | `screen -r experiment` |
| Detach from screen | `Ctrl+A` then `D` |
| Check screen sessions | `screen -ls` |
| Check GPU | `nvidia-smi` |

---

## Setup

```bash
cd re_implementation
bash setup_env.sh          # creates conda env or venv + installs deps
```

Activate the environment:
```bash
# If conda was used:
conda activate ./.conda_env
# Otherwise:
source .venv/bin/activate
```

## Training

```bash
python train.py \
    --data-dir ../data \
    --save-dir ./checkpoints \
    --epochs 10 \
    --batch-size 32 \
    --accum-steps 4 \
    --lr 1e-3 \
    --vocab-size 32000 \
    --amp
```

On the first run the BPE tokenizer is trained and cached in `checkpoints/sp/`.
Encoded token IDs are also cached as `.pt` files in `checkpoints/sp/cache/` —
subsequent runs skip the encoding step entirely (~3-4 min saved).

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | 32,000 | BPE vocabulary size |
| `--emb-dim` | 256 | Embedding dimension |
| `--hidden-size` | 512 | LSTM hidden size |
| `--batch-size` | 64 | Micro-batch size per forward pass |
| `--accum-steps` | 1 | Gradient accumulation (effective batch = batch × accum) |
| `--epochs` | 15 | Number of epochs (mutually exclusive with `--max-steps`) |
| `--max-steps` | — | Total optimizer steps |
| `--patience` | 5 | Early stopping patience (validation rounds) |
| `--eval-every` | — | Validate every N steps (default: once per epoch) |
| `--save-every` | — | Numbered checkpoint every N steps |
| `--log-every` | 100 | Print training stats every N steps |
| `--amp` | off | Mixed-precision training (needs CUDA) |
| `--resume` | — | Resume from a checkpoint `.pt` |
| `--max-train-lines` | 0 | Truncate training data (0 = all). Useful for smoke tests |

### Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/best.pt` | Best model by validation loss (updated at each `--eval-every`) |
| `checkpoints/last.pt` | Most recent checkpoint |
| `checkpoints/step_N.pt` | Numbered checkpoints (every `--save-every` steps) |

---

## Evaluation

### Full evaluation with beam search (recommended)
```bash
python evaluate.py \
    --data-dir ../data \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --beam-size 5
```

### Quick greedy evaluation (much faster, slightly lower scores)
```bash
python evaluate.py \
    --data-dir ../data \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --batch-decode
```

### Evaluate a specific numbered checkpoint
```bash
python evaluate.py \
    --data-dir ../data \
    --checkpoint ./checkpoints/step_10000.pt \
    --output-dir ./testing_res_step10k \
    --batch-decode
```

### Limit to N test samples (quick sanity check)
```bash
python evaluate.py \
    --data-dir ../data \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --batch-decode \
    --max-samples 100
```

### Output files (`testing_res/`)

| File | Description |
|------|-------------|
| `predictions.txt` | Predicted vs reference translations |
| `metrics.txt` / `metrics.json` | BLEU-1/2/3/4, ROUGE-1/2/L scores |
| `comparison_report.txt` | Side-by-side comparison with the paper's results |
| `comparison.json` | Machine-readable comparison data |
| `comparison_chart.png` | Bar chart: ours vs paper |
| `bleu_scores.png` | BLEU bar chart |
| `rouge_scores.png` | ROUGE bar chart (P / R / F) |
| `length_distribution.png` | Predicted vs reference length histogram |
| `bleu4_distribution.png` | Per-sentence BLEU-4 histogram |

---

## Full Automated Experiment

The `run_experiment.sh` script automates everything end-to-end:

```bash
# Run in a detached screen session (safe to close terminal)
screen -S experiment bash run_experiment.sh
```

This will:
1. **Train** the model for 10 epochs with paper-matched hyperparameters
2. **Evaluate** with beam search on the full test set (newstest2014)
3. **Generate** comparison report against the paper's published metrics
4. **Save** all results, plots, and checkpoints

### Monitoring

```bash
# Quick check — last few log lines
tail -20 experiment_full.log

# Live follow
tail -f experiment_full.log     # Ctrl+C to stop

# Reattach to screen (see full live output)
screen -r experiment            # Ctrl+A, D to detach
```

### Paper baseline (Mini-Former)

| Metric | Paper |
|--------|-------|
| BLEU-1 | 0.42 |
| BLEU-2 | 0.22 |
| BLEU-3 | 0.14 |
| BLEU-4 | 0.09 |
| ROUGE-1 (P/R/F) | 0.59 / 0.44 / 0.49 |
| ROUGE-2 (P/R/F) | 0.28 / 0.23 / 0.25 |
| ROUGE-L (P/R/F) | 0.57 / 0.42 / 0.46 |

---

## Project structure

```
re_implementation/
  data.py              — SentencePiece BPE tokenizer, dataset, dataloader, caching
  model.py             — Seq2Seq + Bahdanau attention (shared embedding)
  train.py             — CLI training loop (step-based, gradient accumulation)
  evaluate.py          — CLI evaluation + metric plots + paper comparison report
  run_experiment.sh    — Automated full experiment (train → evaluate → report)
  requirements.txt     — Python dependencies
  setup_env.sh         — Environment setup script
  README.md            — This file
```
