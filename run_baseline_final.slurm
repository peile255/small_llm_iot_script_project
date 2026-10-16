#!/bin/bash
#SBATCH --job-name=baseline_final
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/baseline_final_%j.out
#SBATCH --error=outputs/baseline_final_%j.err

set -euo pipefail

cd ~/small_llm_iot_script_project

mkdir -p outputs

module purge
module load python/3.10.13-7ad3v37
source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== HOST ==="
hostname

echo "=== VERIFY FROZEN DATA ==="
(
    cd data/processed/final_frozen
    sha256sum -c SHA256SUMS.txt
)

echo "=== RUN FINAL BASELINE ==="

python run_baseline.py \
  --config configs/experiment_final.yaml

echo "=== FINAL BASELINE METRICS ==="
cat outputs/baseline/metrics.json

echo "=== DONE ==="
