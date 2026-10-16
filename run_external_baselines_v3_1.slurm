#!/bin/bash
#SBATCH --job-name=ext_base31
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/paper_v3/external_baselines_v3_1_%j.out
#SBATCH --error=outputs/paper_v3/external_baselines_v3_1_%j.err

set -euo pipefail

cd ~/small_llm_iot_script_project

module purge
module load python/3.10.13-7ad3v37
source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== HOST ==="
hostname

echo "=== VERIFY INTERNAL FREEZE ==="
(
    cd data/processed/final_frozen
    sha256sum -c SHA256SUMS.txt
)

echo "=== VERIFY EXTERNAL FREEZE ==="
(
    cd outputs/paper_v3/frozen_external_v1
    sha256sum -c SHA256SUMS.txt
)

echo "=== RUN SHARED-FEATURE EXTERNAL BASELINES ==="

python scripts/run_external_baselines_v3_1.py

echo "=== SUMMARY ==="

cat \
outputs/paper_v3/external_baselines_v3_1/external_baseline_summary.csv

echo "=== DONE ==="
