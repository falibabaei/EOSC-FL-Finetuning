#!/bin/bash
#SBATCH --job-name=eosc_fl_llm
#SBATCH --output=logs/eosc_fl_%j.out
#SBATCH --error=logs/eosc_fl_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

module purge
module load python/3.12
module load cuda/12.1

source /home/se1131/tt9c_uncertainty_quantification/venv/bin/activate

mkdir -p logs

echo "=== EOSC FL LLM Fine-tuning Simulation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'none')"
echo "Start: $(date)"
echo ""

WORKSPACE="/tmp/eosc_fl_workspace_${SLURM_JOB_ID}"

python /home/se1131/tt9c_uncertainty_quantification/FL_llm_finetuning/run_fl_simulation.py \
    --rounds 10 \
    --clients 4 \
    --threads 4 \
    --workspace "$WORKSPACE"

echo ""
echo "End: $(date)"
echo "Done."
