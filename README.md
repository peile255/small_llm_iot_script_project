# Small-LLM Malicious Script Identification for IoT Security

Reproducible static-only experiment project for the manuscript:

**Analyzing the Efficacy of Small-Scale LLMs in Malicious Script Identification for IoT Security**

## Safety Model

All dataset scripts are treated as untrusted inert text.

- Dataset scripts are never executed.
- Raw malware samples are not bundled with this repository.
- No dataset sample is passed to `pwsh`, `powershell.exe`, `bash`, `cmd`, `eval`, or `exec`.
- Malware data should be handled in an isolated research environment and according to dataset licenses and institutional policies.

## Public Data Sources

### Primary dataset

MPSD:

https://github.com/das-lab/mpsd

Associated publication:

Fang, Zhou, Huang, *Neurocomputing* 448 (2021), 30--39.

DOI:

10.1016/j.neucom.2021.03.117

### External malicious-source stress-test dataset

https://github.com/Fa2y/Malicious-PowerShell-Dataset

This dataset contains malicious samples only. Evaluation on this source should therefore focus on metrics such as recall and false-negative rate.

### Optional secondary labeled dataset

https://huggingface.co/datasets/wwe123/Benign_and_malicious_PowerShell_scripts

This dataset may be used for additional external evaluation after provenance and overlap auditing.

## Final Experimental Pipeline

The final paper workflow is:

1. Download the public MPSD data.
2. Prepare the primary dataset.
3. Remove cross-label exact-hash conflicts.
4. Remove exact duplicates using SHA-256.
5. Normalize scripts for similarity analysis.
6. Construct character-level TF-IDF representations.
7. Build near-duplicate groups using cosine similarity >= 0.90.
8. Exclude development-contaminated groups from the final test set.
9. Build leakage-controlled train/validation/test partitions.
10. Freeze and checksum the final split.
11. Run the conventional baseline.
12. Run the five final Ollama-based local LLMs.
13. Generate per-sample predictions and metrics.
14. Run statistical and long-script analyses.
15. Evaluate selective hybrid verification.
16. Generate publication figures.
17. Generate LaTeX tables.

## Final Dataset Preparation

Generate the leakage-controlled final split with:

```bash
python scripts/prepare_final_split.py
```

The final split procedure includes:

- SHA-256 hashing;
- removal of cross-label exact-hash conflicts;
- exact duplicate removal;
- lowercasing and whitespace normalization for similarity analysis;
- character-level TF-IDF with 4--5 character n-grams;
- cosine-similarity near-duplicate detection;
- near-duplicate threshold of 0.90;
- group-aware splitting;
- exclusion of groups containing development samples from the final test set;
- SHA-256 overlap checks;
- near-duplicate group overlap checks.

Frozen final data are stored under:

```text
data/processed/final_frozen/
```

Expected frozen files include:

```text
final_train.csv
final_val.csv
final_test.csv
final_split_summary.json
SHA256SUMS.txt
```

Verify frozen data before running final experiments:

```bash
cd data/processed/final_frozen
sha256sum -c SHA256SUMS.txt
cd ../../..
```

## Conventional Baseline

The final conventional baseline uses character-level TF-IDF features and logistic regression.

Configuration:

```text
configs/experiment_final.yaml
```

Final SLURM script:

```text
run_baseline_final.slurm
```

Run:

```bash
sbatch run_baseline_final.slurm
```

Important baseline settings:

```text
character n-grams: 3--5
maximum TF-IDF features: 200000
logistic regression C: 2.0
class_weight: balanced
seed: 42
```

## Final Local LLMs

The final study evaluates:

- Mistral-7B-Instruct-v0.3
- Phi-4
- Gemma-3-27B
- QwQ-32B
- DeepSeek-R1-32B

The final LLM configuration is:

```text
configs/ollama_final.yaml
```

The final inference program is:

```text
run_ollama.py
```

The final SLURM script is:

```text
run_ollama_final.slurm
```

Final generation settings:

```text
temperature: 0
seed: 42
num_ctx: 8192
num_predict: 2048
max_input_chars: 6000
output_format: json
```

Scripts longer than 6000 characters use a fixed-budget head-tail representation:

```text
script head
...[TRUNCATED]...
script tail
```

The value `num_ctx=8192` is the configured Ollama context size for the experiment.

## Running the Final LLM Experiments

Run each model separately using `MODEL_KEY`.

```bash
MODEL_KEY=mistral_7b sbatch run_ollama_final.slurm
MODEL_KEY=phi4 sbatch run_ollama_final.slurm
MODEL_KEY=gemma3_27b sbatch run_ollama_final.slurm
MODEL_KEY=qwq_32b sbatch run_ollama_final.slurm
MODEL_KEY=deepseek_r1_32b sbatch run_ollama_final.slurm
```

The final LLM jobs were configured for:

```text
GPU: NVIDIA A100 40 GB
GPUs per job: 1
CPU cores: 4
RAM: 32 GB
Backend: Ollama
Container runtime: Apptainer
```

## Final Output Structure

Final experiment outputs are stored under:

```text
outputs/final/
```

Model directories:

```text
outputs/final/baseline/
outputs/final/mistral_7b/
outputs/final/phi4/
outputs/final/gemma3_27b/
outputs/final/qwq_32b/
outputs/final/deepseek_r1_32b/
```

Each final LLM directory contains:

```text
test_predictions.csv
metrics.json
```

Per-sample LLM prediction files preserve information including:

- true label;
- predicted label;
- structured-output validity;
- confidence;
- API status;
- truncation status;
- original character count;
- transmitted character count;
- prompt/evaluation token statistics;
- latency;
- raw model output.

## Final Result Files

Derived result tables are stored as:

```text
outputs/final/main_model_results.csv
outputs/final/long_script_analysis.csv
outputs/final/hybrid_results.csv
outputs/final/bootstrap_ci_summary.csv
outputs/final/paired_bootstrap_f1.csv
```

## Statistical Analysis

The final analysis includes:

- accuracy;
- precision;
- recall;
- F1;
- Matthews correlation coefficient;
- false-positive and false-negative counts;
- invalid-output rate;
- inference latency and throughput;
- bootstrap 95% confidence intervals;
- exact McNemar paired tests;
- paired bootstrap F1 differences;
- script-length-stratified analysis;
- complementary-error analysis;
- selective hybrid analysis.

## Long-Script Analysis

The final test set is stratified into:

```text
<= 6000 characters
> 6000 characters
```

For LLM inference, scripts in the second group are represented using the 6000-character head-tail strategy.

The distinction is based on the original script length. The conventional baseline itself is not described as using this LLM truncation mechanism.

## Selective Hybrid Verification

A selective second-stage verification strategy is evaluated.

Decision rule:

```text
if baseline prediction == negative
and original script length > 6000 characters:
    use the LLM prediction for second-stage verification
else:
    retain the baseline prediction
```

Hybrid variants are evaluated using:

- Phi-4
- DeepSeek-R1-32B
- QwQ-32B

This strategy is intended to study whether complementary LLM errors can reduce false negatives without requiring LLM inference for every sample.

## Publication Figures

Figures are stored under:

```text
outputs/final/figures/
```

Generated figures:

```text
fig1_overall_f1.pdf
fig1_overall_f1.png
fig2_precision_recall.pdf
fig2_precision_recall.png
fig3_length_f1.pdf
fig3_length_f1.png
fig4_hybrid_fp_fn.pdf
fig4_hybrid_fp_fn.png
```

## LaTeX Tables

Publication-ready LaTeX tables are stored under:

```text
outputs/final/paper/
```

Generated tables:

```text
table1_overall.tex
table2_length.tex
table3_hybrid.tex
```

## Environment Setup

Create and activate the Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Important analysis dependencies include:

```text
pandas
numpy
scikit-learn
scipy
matplotlib
statsmodels
```

## Tests

Run unit tests with:

```bash
pytest
```

## Reproducibility Notes

- Final LLM random seed: 42.
- LLM temperature: 0.
- All final classifiers are evaluated on the same frozen test manifest.
- Frozen data integrity is checked using SHA-256 checksums.
- Exact duplicates are removed before splitting.
- Near-duplicate scripts are grouped using character-level TF-IDF cosine similarity.
- Near-duplicate groups are kept disjoint across final partitions.
- Development-contaminated groups are excluded from the final test set.
- Raw LLM outputs are retained for audit.
- Invalid structured outputs are explicitly recorded.
- LLM input truncation uses a fixed 6000-character head-tail strategy.
- Dataset scripts are never executed.

## Data Provenance

Dataset provenance is recorded in:

```text
DATA_PROVENANCE.json
```

Raw malware samples are intentionally not bundled with this repository.

## Complete Reproduction Sequence

```text
1. Download public data
2. Prepare primary data
3. Build leakage-controlled final split
4. Verify SHA256SUMS
5. Run final baseline
6. Run five final Ollama LLMs
7. Run final statistical analysis
8. Generate publication figures
9. Generate LaTeX tables
```

Paper metrics should be generated from the files under `outputs/final/` rather than manually hard-coded.
