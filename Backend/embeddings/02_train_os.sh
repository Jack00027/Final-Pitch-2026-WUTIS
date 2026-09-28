#!/bin/bash
#SBATCH --job-name=osbert-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# Full OS-BERT training sweep across all quarters in data_wutis/.
#
# Before submitting, make sure logs/ exists:
#   mkdir -p logs
#
# Submit with:
#   sbatch 02_train_os.sh                      # full sweep
#   sbatch 02_train_os.sh --first-only         # just the first quarter (dress rehearsal)
#
# Watch progress:
#   squeue -u $USER
#   tail -f logs/osbert-train-<jobid>.out

set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "OS-BERT training on $(hostname)"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "═══════════════════════════════════════════════════════════════"

# Show what GPU we got
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Load env
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

cd ~/WUTIS

# Optional: if first arg is --first-only, run just quarter 1 as a dress rehearsal.
# Otherwise run the full sweep (skip_existing=True means safe to resubmit).
if [[ "${1:-}" == "--first-only" ]]; then
    echo ""
    echo "→ DRESS REHEARSAL MODE: stopping after first quarter"
    echo ""
    python <<'PYEOF'
import sys
sys.argv = ['OS_BERT_training.py']
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("OS_BERT_training", "OS_BERT_training.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

cfg = bt.default_config()
import random, numpy as np, torch
random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])

print(f"device     : {cfg['device']}")
print(f"seq_dir    : {cfg['seq_dir']}")
print(f"emb_dir    : {cfg['emb_dir']}")
print(f"model_dir  : {cfg['model_dir']}")

files = sorted(Path(cfg["seq_dir"]).glob("q_*.parquet"))
print(f"\nProcessing only the FIRST of {len(files)} quarters\n")
bt.process_quarter(files[0], cfg)
print("\nDress rehearsal complete.")
PYEOF
else
    echo ""
    echo "→ FULL SWEEP MODE: processing all quarters"
    echo ""
    python OS_BERT_training.py
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
