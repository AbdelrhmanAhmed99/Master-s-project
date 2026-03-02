# Multilingual Seq2Seq Machine Translation

Many → **English** translation using a Seq2Seq + Bahdanau Attention model
(PyTorch).

Trains a single model on **5 source languages** simultaneously:

| Pair | Source | Target |
|------|--------|--------|
| de → en | German | English |
| fr → en | French | English |
| cs → en | Czech | English |
| ru → en | Russian | English |
| es → en | Spanish | English |

Uses a **shared SentencePiece BPE tokenizer** (32K vocab, character_coverage
0.9999 for Latin + Cyrillic) trained on all source + target data.

**Data sources:** Europarl v10, News Commentary v18, WikiMatrix v1,
Tilde Model Corpus — totalling ~27 M sentence pairs.

**Test set:** 1% stratified sample from the training corpus, saved as fixed
TSV files (one per language pair) for reproducible evaluation.

---

## Quick Reference

| Task | Command |
|------|---------|
| Run full experiment (train + eval) | `screen -S experiment bash run_experiment.sh` |
| Evaluate latest checkpoint | `python evaluate.py --data-dir /workspace/Datasets --checkpoint ./checkpoints/best.pt --output-dir ./testing_res --beam-size 5` |
| Quick greedy eval (fast) | `python evaluate.py --data-dir /workspace/Datasets --checkpoint ./checkpoints/best.pt --output-dir ./testing_res --batch-decode` |
| Check training progress | `tail -20 experiment.log` |
| Live training log | `tail -f experiment.log` |
| Reattach to screen | `screen -r experiment` |
| Detach from screen | `Ctrl+A` then `D` |
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
    --data-dir /workspace/Datasets \
    --save-dir ./checkpoints \
    --epochs 10 \
    --batch-size 32 \
    --accum-steps 4 \
    --lr 1e-3 \
    --vocab-size 32000 \
    --amp
```

On the first run:
1. All corpora are loaded and a 3-way stratified split is created (train 98% / val 1% / test 1%)
2. The test split is saved as fixed TSV files in `<data-dir>/test_sets/` and reused on all subsequent runs
3. The shared BPE tokenizer is trained and cached in `checkpoints/sp/`
4. Encoded token IDs are cached as `.pt` files in `checkpoints/sp/cache/`

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | *(required)* | Root dataset directory (e.g. `/workspace/Datasets`) |
| `--vocab-size` | 32,000 | BPE vocabulary size |
| `--emb-dim` | 256 | Embedding dimension |
| `--hidden-size` | 512 | LSTM hidden size |
| `--batch-size` | 64 | Micro-batch size per forward pass |
| `--accum-steps` | 1 | Gradient accumulation (effective batch = batch × accum) |
| `--epochs` | 15 | Number of epochs (mutually exclusive with `--max-steps`) |
| `--max-steps` | — | Total optimizer steps |
| `--val-ratio` | 0.01 | Fraction held out for validation per (lang, dataset) |
| `--test-ratio` | 0.01 | Fraction held out for testing per (lang, dataset) |
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
| `checkpoints/best.pt` | Best model by validation loss |
| `checkpoints/last.pt` | Most recent checkpoint |
| `checkpoints/step_N.pt` | Numbered checkpoints (every `--save-every` steps) |

---

## Evaluation

Evaluation produces **combined** metrics (across all 5 language pairs) and
**per-language** breakdowns (BLEU-1..4, ROUGE-1/2/L) with diagnostic plots.

### Full evaluation with beam search (recommended)
```bash
python evaluate.py \
    --data-dir /workspace/Datasets \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --beam-size 5
```

### Quick greedy evaluation (much faster, slightly lower scores)
```bash
python evaluate.py \
    --data-dir /workspace/Datasets \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --batch-decode
```

### Limit to N test samples (quick sanity check)
```bash
python evaluate.py \
    --data-dir /workspace/Datasets \
    --checkpoint ./checkpoints/best.pt \
    --output-dir ./testing_res \
    --batch-decode \
    --max-samples 100
```

### Output files (`testing_res/`)

| File | Description |
|------|-------------|
| `predictions.txt` | Predicted vs reference translations (tagged by language pair) |
| `metrics.txt` | Combined + per-language BLEU & ROUGE table |
| `metrics.json` | Machine-readable combined + per-language metrics |
| `bleu_scores.png` | BLEU bar chart (combined) |
| `bleu_per_language.png` | Grouped BLEU-1..4 per language pair |
| `rouge_scores.png` | ROUGE P/R/F bar chart (combined) |
| `rouge_per_language.png` | ROUGE-L F per language pair |
| `length_distribution.png` | Predicted vs reference length histogram |
| `bleu4_distribution.png` | Per-sentence BLEU-4 histogram |

---

## Full Automated Experiment

```bash
# Run in a detached screen session (safe to close terminal)
screen -S experiment bash run_experiment.sh
```

This will:
1. **Train** the model on all 5 language pairs
2. **Evaluate** with beam search on the fixed test set
3. **Generate** per-language and combined metric reports + plots
4. **Save** all results, plots, and checkpoints

### Monitoring
```bash
tail -20 experiment.log        # Quick check
tail -f experiment.log         # Live follow (Ctrl+C to stop)
screen -r experiment           # Reattach (Ctrl+A, D to detach)
```

---

## Data Pipeline

All data processing is handled by `data.py`. The pipeline:

1. **Loads** 4 corpus families (Europarl v10, News Commentary v18, WikiMatrix v1, Tilde TMX)
2. **Normalises** column order (auto-detects swapped `en-<lang>` files)
3. **Filters** WikiMatrix pairs below score 1.05
4. **Splits** into train / val / test via stratified sampling per (language, dataset)
5. **Saves** the test split as fixed TSV files (reused on all subsequent runs)
6. **Trains** a shared SentencePiece BPE tokenizer on training data
7. **Encodes** all splits and caches as `.pt` files

### Data summary (~27 M pairs total)

| Language | Europarl | News Commentary | WikiMatrix | Tilde | Total |
|----------|----------|-----------------|------------|-------|-------|
| de → en | 1.8 M | 438 K | 1.0 M | 5.2 M | 8.5 M |
| fr → en | 1.9 M | 407 K | — | 5.1 M | 7.5 M |
| cs → en | 644 K | 265 K | 336 K | 2.1 M | 3.3 M |
| ru → en | — | 378 K | 1.2 M | 34 K | 1.6 M |
| es → en | 1.9 M | 500 K | — | 3.8 M | 6.2 M |

---

## Project structure

```
re_implementation/
  data.py              — Multilingual data pipeline, SentencePiece, caching
  model.py             — Seq2Seq + Bahdanau attention (shared embedding)
  train.py             — CLI training loop (step-based, gradient accumulation)
  evaluate.py          — CLI evaluation + per-language metrics + plots
  run_experiment.sh    — Automated full experiment (train → evaluate → report)
  requirements.txt     — Python dependencies
  setup_env.sh         — Environment setup script
  README.md            — This file
```
