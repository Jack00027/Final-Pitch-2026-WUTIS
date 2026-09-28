#!/bin/bash
#SBATCH --job-name=osbert-smoke
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# 01_smoke_test_os.sh
# Submit with:  sbatch 01_smoke_test_os.sh
#
# What it does:
#   - Asks SLURM for 1 A30 GPU on the `test` partition for 10 minutes.
#   - Loads miniconda + activates the `psbert` env.
#   - Runs a tiny PyTorch script that confirms CUDA works on the GPU.
#
# If this exits with status 0 and the .out file shows "GPU matmul OK",
# you're ready to run real training.

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Running on node:  $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Allocated CPUs:   $SLURM_CPUS_PER_TASK"
echo "Started at:       $(date)"
echo "Working dir:      $(pwd)"
echo "=========================================="

# ---------- Activate the env ----------
module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

# ---------- Show the GPU we got ----------
echo ""
echo ">>> nvidia-smi"
nvidia-smi

# ---------- Tiny PyTorch sanity check ----------
echo ""
echo ">>> PyTorch / CUDA test"
python <<'PY'
import torch, transformers, time
print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"CUDA built:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available — something is wrong.")

dev = torch.device("cuda:0")
print(f"Device 0:     {torch.cuda.get_device_name(0)}")
print(f"Compute cap:  {torch.cuda.get_device_capability(0)}")

# A real GPU computation
torch.cuda.synchronize()
t0 = time.time()
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
elapsed = time.time() - t0
flops = 20 * 2 * (4096 ** 3) / elapsed / 1e12
print(f"GPU matmul OK ({elapsed:.2f}s, ~{flops:.1f} TFLOPS bf16)")
print(f"Peak VRAM:    {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY

echo ""
echo "Finished at: $(date)"
echo "=========================================="
