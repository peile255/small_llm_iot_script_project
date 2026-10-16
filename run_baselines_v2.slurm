#!/bin/bash
#SBATCH --job-name=baseline_v2
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/paper_v2/baseline_v2_%j.out
#SBATCH --error=outputs/paper_v2/baseline_v2_%j.err

set -euo pipefail

cd ~/small_llm_iot_script_project

module purge
module load python/3.10.13-7ad3v37
source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== VERIFY FROZEN DATA ==="
(
  cd data/processed/final_frozen
  sha256sum -c SHA256SUMS.txt
)

echo "=== RUN BASELINE V2 ==="
python scripts/run_baselines_v2.py

echo "=== SUMMARY ==="
cat outputs/paper_v2/baselines/baseline_v2_summary.csv

echo "=== DONE ==="
