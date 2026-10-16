# Leakage-Controlled Evaluation of Local LLMs for Malicious PowerShell Detection

Reproducibility package for the study:

**Beyond In-Distribution Accuracy: Leakage-Controlled External Evaluation of Local LLMs for Malicious PowerShell Detection**

This repository contains the code, frozen manifests, per-sample predictions, statistical outputs, and robustness analyses used in the study. The main comparison covers:

- conventional sparse-text classifiers;
- a pretrained discriminative code model (CodeBERT); and
- five local generative LLMs.

The central result is that **model ordering is distribution dependent**: sparse lexical classifiers are strongest on the leakage-controlled internal test set, whereas CodeBERT and several local LLMs improve substantially under contamination-controlled external evaluation.

> **Safety:** PowerShell files are treated only as inert text. Dataset scripts are never executed by the experimental pipeline.

---

## Repository Layout

```text
.
├── configs/
│   ├── experiment_final.yaml
│   ├── ollama_final.yaml
│   ├── v2/baselines_v2.yaml
│   └── v4/codebert_v4.yaml
├── data/
│   └── processed/final_frozen/
├── outputs/
│   ├── paper_v2/frozen_results_v2/
│   ├── paper_v3/
│   │   ├── frozen_external_v1/
│   │   ├── frozen_external_llm_subset_v1/
│   │   └── frozen_external_results_v1/
│   └── paper_v4/
│       ├── frozen_distribution_shift_v1/
│       ├── frozen_codebert_v1/
│       ├── frozen_codebert_statistics_v1/
│       └── frozen_threshold_sensitivity_v1/
├── scripts/
├── src/
├── requirements.txt
└── SHA256SUMS.txt
```

The `frozen_*` directories are the result snapshots used for manuscript integration. They should be treated as read-only.

---

## Research Questions

The final study addresses six questions:

1. **RQ1 — Internal comparison:** How do conventional sparse-text classifiers compare with local LLMs on a leakage-controlled frozen test set?
2. **RQ2 — Long scripts:** How does performance differ for scripts exceeding 6,000 characters?
3. **RQ3 — Selective verification:** Can an exploratory hybrid rule exploit complementary LLM errors while invoking an LLM only for selected cases?
4. **RQ4 — External validation:** Does the internal model ordering persist on a contamination-controlled external corpus?
5. **RQ5 — Audit robustness:** Is the external conventional-model ordering sensitive to the near-duplicate exclusion threshold?
6. **RQ6 — Lexical distance:** How is distance from the frozen training distribution associated with relative external performance?

---

## Model Families

### Conventional classifiers

The primary sparse classifier is a character- and word-level TF-IDF feature union with logistic regression.

Additional robustness baselines are:

- character TF-IDF + logistic regression;
- character TF-IDF + linear SVM;
- word TF-IDF + linear SVM.

Relevant files:

```text
configs/v2/baselines_v2.yaml
scripts/run_baselines_v2.py
run_baselines_v2.slurm
scripts/run_external_baselines_v3_1.py
run_external_baselines_v3_1.slurm
```

### CodeBERT

The pretrained discriminative baseline uses:

```text
microsoft/codebert-base
```

Relevant files:

```text
configs/v4/codebert_v4.yaml
scripts/run_codebert_v4.py
outputs/paper_v4/frozen_codebert_v1/
outputs/paper_v4/frozen_codebert_statistics_v1/
```

The fine-tuned checkpoint weights are intentionally not included in this GitHub package because of repository-size constraints. The frozen prediction files and evaluation summaries are included, and the model can be reproduced from the configuration and training script. See:

```text
outputs/paper_v4/frozen_codebert_v1/WEIGHTS_NOT_INCLUDED.txt
```

### Local LLMs

The final LLM family contains:

- Mistral-7B;
- Phi-4;
- Gemma-3-27B;
- QwQ-32B;
- DeepSeek-R1-32B.

Final inference settings are recorded in:

```text
configs/ollama_final.yaml
```

External LLM execution is implemented in:

```text
scripts/run_external_llm_v3.py
```

The frozen external LLM predictions are stored under:

```text
outputs/paper_v3/frozen_external_results_v1/external_llm_metrics/
```

---

## Frozen Internal Split

The primary corpus begins with 8,518 PowerShell files. The frozen pipeline removes cross-label exact-hash conflicts, exact duplicates, and leakage-inducing near-duplicate groups before constructing the final split.

Frozen split sizes:

| Partition | n |
|---|---:|
| Train | 4,322 |
| Validation | 609 |
| Test | 1,265 |

The frozen test set contains 1,049 benign and 216 malicious scripts.

Files:

```text
data/processed/final_frozen/final_train.csv
data/processed/final_frozen/final_val.csv
data/processed/final_frozen/final_test.csv
data/processed/final_frozen/final_split_summary.json
```

The split-construction script is:

```bash
python scripts/prepare_final_split.py
```

The final split uses exact hashing plus character-boundary TF-IDF near-duplicate grouping with cosine-similarity threshold 0.90.

---

## External Evaluation and Contamination Audit

The external source used for the final contamination-controlled evaluation is:

```text
wwe123/Benign_and_malicious_PowerShell_scripts
```

The downloaded source contains 30,595 rows. After conflict removal, exact deduplication, exact overlap removal against the primary corpus, and near-duplicate exclusion, the final audited corpus contains:

```text
19,632 samples
9,324 benign
10,308 malicious
```

The main audit scripts are:

```text
scripts/audit_external_exact_v3.py
scripts/build_external_exact_clean_v3.py
scripts/audit_external_neardup_v3.py
```

Frozen audit outputs are under:

```text
outputs/paper_v3/frozen_external_v1/
```

The balanced external subset used for paired LLM/CodeBERT comparison contains 2,000 samples:

```text
1,000 benign
1,000 malicious
```

Its frozen manifest is:

```text
outputs/paper_v3/frozen_external_llm_subset_v1/external_llm_subset_2000.csv
```

---

## Main Frozen Results

### Internal conventional robustness and paired statistics

```text
outputs/paper_v2/frozen_results_v2/
```

Key files include:

```text
baseline_v2_summary.csv
group_bootstrap_ci.csv
group_paired_delta_f1.csv
mcnemar_holm_familywise.csv
long_group_bootstrap_ci.csv
long_group_paired_delta_f1.csv
long_mcnemar_holm.csv
operational_results.csv
```

### External predictions and paired statistics

```text
outputs/paper_v3/frozen_external_results_v1/
```

This directory includes:

- full-audited-corpus predictions for four conventional classifiers;
- balanced-2,000 predictions for the five local LLMs;
- paired F1 bootstrap analyses;
- exact McNemar tests;
- long-script stratified analyses.

### CodeBERT

```text
outputs/paper_v4/frozen_codebert_v1/
outputs/paper_v4/frozen_codebert_statistics_v1/
```

These include internal, validation, and external per-sample predictions plus paired statistical comparisons with the primary sparse classifier.

### Lexical distribution-shift analysis

```text
outputs/paper_v4/frozen_distribution_shift_v1/
```

Relevant scripts:

```text
scripts/analyze_distribution_shift_v4.py
scripts/analyze_similarity_bin_performance_v4.py
scripts/analyze_distance_q1_statistics_v4.py
scripts/analyze_class_similarity_margin_v4.py
```

### Near-duplicate threshold sensitivity

```text
outputs/paper_v4/frozen_threshold_sensitivity_v1/
```

Relevant scripts:

```text
scripts/analyze_neardup_threshold_sensitivity_v4.py
scripts/run_external_baselines_threshold095_v4.py
scripts/check_threshold095_identity_v4.py
scripts/finalize_threshold_sensitivity_v4.py
```

The evaluated exclusion thresholds are:

```text
0.80, 0.85, 0.90, 0.95
```

The conventional-model ranking is unchanged across all four thresholds.

---

## Key Frozen Results

### Internal test set

| Model | F1 | MCC |
|---|---:|---:|
| LR Char+Word | 0.943 | 0.932 |
| LR Char | 0.934 | 0.921 |
| SVM Char | 0.941 | 0.930 |
| SVM Word | 0.943 | 0.932 |
| CodeBERT | 0.935 | 0.922 |
| Mistral-7B | 0.792 | 0.751 |
| Phi-4 | 0.896 | 0.876 |
| Gemma-3-27B | 0.608 | 0.565 |
| QwQ-32B | 0.775 | 0.741 |
| DeepSeek-R1-32B | 0.857 | 0.833 |

CodeBERT's paired internal F1 difference from the primary LR Char+Word classifier is not statistically clear.

### Balanced external subset (n = 2,000)

| Model | F1 | MCC |
|---|---:|---:|
| LR Char+Word | 0.807 | 0.714 |
| LR Char | 0.813 | — |
| SVM Char | 0.844 | — |
| SVM Word | 0.813 | — |
| CodeBERT | 0.875 | 0.792 |
| Mistral-7B | 0.897 | 0.804 |
| Phi-4 | 0.943 | 0.891 |
| Gemma-3-27B | 0.815 | 0.604 |
| QwQ-32B | 0.909 | 0.823 |
| DeepSeek-R1-32B | 0.916 | 0.840 |

Several local LLMs and CodeBERT improve clearly over the primary sparse classifier under external evaluation.

### Lexically farthest malicious subset

RQ6 uses the 200 external malicious scripts in the farthest nearest-training-similarity quintile.

| Model | Recall |
|---|---:|
| LR Char+Word | 0.045 |
| LR Char | 0.045 |
| SVM Word | 0.070 |
| SVM Char | 0.170 |
| CodeBERT | 0.415 |
| Mistral-7B | 0.625 |
| DeepSeek-R1-32B | 0.625 |
| QwQ-32B | 0.635 |
| Phi-4 | 0.730 |
| Gemma-3-27B | 0.870 |

This analysis is descriptive of association with lexical training-set distance. It should not be interpreted as establishing a causal mechanism.

---

## Environment Setup

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core analysis dependencies include NumPy, pandas, SciPy, scikit-learn, matplotlib, and statsmodels. CodeBERT additionally requires the Hugging Face/PyTorch stack specified by the project environment used for `scripts/run_codebert_v4.py`.

Local LLM inference requires Ollama and locally available model weights corresponding to the configured model names.

---

## Recommended Reproduction Paths

### A. Verify the frozen package only

From the repository root:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

On Linux, the equivalent is:

```bash
sha256sum -c SHA256SUMS.txt
```

This checks the integrity of the GitHub reproducibility package.

### B. Recompute statistical analyses from frozen predictions

The fastest reviewer path is to use the included per-sample predictions and rerun the analysis scripts rather than repeating all LLM inference.

Examples:

```bash
python scripts/analyze_statistics_v2.py
python scripts/analyze_external_statistics_v3.py
python scripts/analyze_codebert_statistics_v4.py
python scripts/analyze_distribution_shift_v4.py
python scripts/analyze_similarity_bin_performance_v4.py
python scripts/analyze_distance_q1_statistics_v4.py
python scripts/finalize_threshold_sensitivity_v4.py
```

Some scripts write to their original experiment output paths. For archival verification, compare regenerated summaries with the corresponding frozen directories rather than overwriting the frozen snapshots.

### C. Reproduce the conventional baselines

Prepare the frozen primary split, then run:

```bash
python scripts/run_baselines_v2.py
```

For the full external conventional evaluation:

```bash
python scripts/run_external_baselines_v3_1.py
```

### D. Reproduce CodeBERT

Use:

```bash
python scripts/run_codebert_v4.py
```

with:

```text
configs/v4/codebert_v4.yaml
```

The original fine-tuned checkpoint is not included, so this path retrains CodeBERT from `microsoft/codebert-base` using the frozen configuration.

### E. Reproduce local-LLM inference

Use the frozen LLM configuration in:

```text
configs/ollama_final.yaml
```

and the available LLM inference scripts. Reproducing the LLM results requires the corresponding local Ollama models and adequate GPU resources.

Because local model binaries are not distributed in this repository, the included frozen per-sample predictions are the recommended artifact for statistical verification.

---

## Statistical Procedures

The frozen analyses use:

- F1 and MCC as primary classification summaries;
- recall for positive-only subsets;
- 10,000 paired bootstrap resamples for paired metric differences;
- exact McNemar tests for paired correctness disagreement;
- Holm correction within predefined comparison families;
- class-stratified lexical-distance analysis;
- long-script stratification using original script length.

McNemar tests quantify paired correctness disagreement and should not be interpreted as direct tests of F1 differences.

---

## Long-Script Analysis

Scripts are partitioned using original character count:

```text
<= 6000 characters
> 6000 characters
```

The local-LLM pipeline uses a fixed maximum input budget. The conventional sparse classifiers process the full script, so differences between short and long scripts are not interpreted as a causal truncation effect.

---

## Selective Hybrid Analysis

A post-hoc selective verifier is evaluated using the rule:

```text
if primary conventional prediction == benign
and original script length > 6000:
    invoke selected LLM verifier
else:
    keep the primary prediction
```

This analysis is exploratory because the rule was developed after final-test error inspection.

---

## Integrity and Frozen Artifacts

Top-level package integrity:

```bash
sha256sum -c SHA256SUMS.txt
```

Several frozen subdirectories also contain their own `SHA256SUMS.txt` files. Those correspond to the original experiment snapshots.

The CodeBERT result directory is an exception: the large checkpoint weights were deliberately omitted from the GitHub package. Its retained prediction and metric files are covered by the top-level repository checksum.

---

## Data and Model Availability

Raw malware samples and large model binaries are not redistributed in this repository.

The repository instead provides:

- source and preparation scripts;
- frozen train/validation/test manifests;
- external audit manifests;
- per-sample model predictions;
- statistical result files;
- frozen summaries and publication tables;
- integrity checksums.

Users should obtain public datasets and model weights from their original providers and comply with their licenses and institutional security policies.

---

## Reproducibility Scope

The repository supports two levels of reproduction:

1. **Artifact-level verification:** recompute metrics/statistics from frozen predictions and verify checksums.
2. **End-to-end rerun:** reacquire public datasets and model weights, rebuild the frozen split/audit pipeline, retrain conventional and CodeBERT models, and rerun local LLM inference.

Exact regeneration of local-LLM outputs may depend on Ollama version, model build, runtime environment, and hardware. The frozen prediction files are therefore provided as the authoritative outputs used in the study.

---

## Citation

If this repository is used in academic work, please cite the associated manuscript once bibliographic details are available.
