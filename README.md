# Preference Tuning Breaks Confidence in Sentiment Analysis with Small Language Models: code and results

Code, tests and generated results for the manuscript "Preference Tuning Breaks Confidence in Sentiment Analysis
with Small Language Models". This copy is anonymised for
peer review.

## What is here

| Path | Content |
|---|---|
| `src/sentalign/` | The package: data construction, the objectives, training, evaluation, statistics |
| `scripts/` | The numbered pipeline: build the data, run the grid, compute every table and figure |
| `tests/` | Unit tests of the objectives, the data pipeline, the protocol and every analysis script |
| `results/` | The generated outputs that the manuscript reads: JSON statistics, LaTeX tables, figures |

Not included: model checkpoints and per-item predictions (about 1 GB for the reported runs).
They will be deposited in a public repository on publication. The corpora are public and the
build scripts download them.

## Install

Python 3.10 or later.

```bash
pip install -r requirements.txt
pip install -e .
```

Training needs one GPU with 24 GB and the training stack:

```bash
pip install -r requirements-train.txt
```

Unsloth is installed separately for your CUDA version. The versions used for every reported
run are PyTorch 2.11 with CUDA 13.0, Transformers 5.5.0, TRL 0.24.0, PEFT 0.20.0,
bitsandbytes 0.50.1 and Unsloth 2026.8.18 on Python 3.12, on one NVIDIA RTX 3090.

## Tests

```bash
python -m pytest -q
```

The tests need no GPU. A few are skipped until the data are built.

## Reproduce

1. Build the data. The scripts download DynaSent v1.1, the re-annotated SST development set
   and GoEmotions, build the splits and the preference pairs, and stop if training and
   evaluation items overlap.

   ```bash
   python scripts/01_build_data.py --out data/build --tau 0.2 --noise 0
   python scripts/01_build_goemo.py --out data/build_goemo
   ```

2. Train and evaluate. The run programme is resumable and runs one arm per subprocess.
   `--dry-run` prints every run with its estimated GPU-hours.

   ```bash
   python scripts/02_run_grid.py --dry-run
   python scripts/02_run_grid.py --studies main confirm landscape ipo-reg beta rpo goemo goemo-smollm3
   ```

   The study `main` also contains exploratory arms that the manuscript does not report.
   `scripts/17_compute_budget.py` defines and counts the 323 reported runs. Two small checks
   were launched with `sentalign train` and its `--set` and `--variant` options: the
   anchor-weight check (`pref_distributional_lambda` at 0.1, 0.3 and 3) and the majority-label
   ablation (`pref_regularizer=hard`).

3. Compute the statistics, tables and figures from the finished runs.

   | Script | Output | In the manuscript |
   |---|---|---|
   | `09_landscape_table.py` | `table_landscape.tex`, `arm_means.json` | Table 1 |
   | `06_regulariser_stats.py`, then `08_regulariser_tables.py` | `regulariser_stats.json`, `table_differences.tex`, `table_soft_vs_hard.tex`, `table_repair_per_set.tex` | Tables 2, 6, 13 |
   | `10_coverage_table.py`, `11_risk_coverage_figure.py` | `table_coverage.tex`, `fig_risk_coverage.pdf` | Table 3, Fig. 5 |
   | `14_lambda_sweep.py` | `table_lambda.tex` | Table 4 |
   | `23_beta_rpo.py` | `table_beta.tex`, `table_rpo.tex` | Tables 5, 7 |
   | `16_goemo_replication.py`, `18_goemo_operating_points.py` | `table_goemo.tex`, `table_goemo_operating.tex` | Tables 8, 9 |
   | `13_agreement_bands.py` | `table_agreement_bands.tex` | Table 10 |
   | `15_deferral_cost.py` | `table_deferral.tex`, `fig_deferral_cost.pdf` | Table 11, Fig. 2 |
   | `21_supplementary_tables.py` | `table_hyperparameters.tex`, `table_vs_sft_per_set.tex`, `table_ipo_reg_smollm3.tex`, `table_placement.tex` | Tables 12, 14, 15, 16 |
   | `20_exact_ordering.py` | `table_exact_ordering.tex` | Table 17 |
   | `19_near_certain_figure.py` | `fig_near_certain.pdf`, `table_near_certain.tex` | Fig. 4, Table 18 |
   | `17_compute_budget.py` | `table_compute.tex` | Table 19 |
   | `22_mechanism.py` | `fig_mechanism.pdf`, `mechanism.json` | Fig. 3 |

   Each script states its command line in its docstring, for example:

   ```bash
   python scripts/06_regulariser_stats.py --runs runs --out results/regulariser_stats.json
   python scripts/08_regulariser_tables.py
   ```

   Fig. 1 is drawn in TikZ inside the manuscript.

## Where the main pieces are

- The anchor: `add_distributional_regulariser` and `distributional_penalty` in
  `src/sentalign/train/po.py`. It wraps a TRL preference trainer and adds the cross-entropy
  term at the last prompt position.
- The objectives and their tests: `src/sentalign/train/objectives.py`, `tests/test_objectives.py`.
- The exact order of the confidence: `src/sentalign/confidence.py`.
- The declared statistical families: `scripts/06_regulariser_stats.py`,
  `scripts/16_goemo_replication.py`, `scripts/23_beta_rpo.py`.
- The run plan: `src/sentalign/plan.py`.

## Notes

- Models: `LiquidAI/LFM2.5-1.2B-Base`, `Qwen/Qwen3.5-2B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`.
- Some scripts and modules serve analyses outside the manuscript (a natural language
  inference study and further exploratory objectives). They are kept because the tests cover
  them and the run programme shares code with them.
- Comments that cite `AUDIT.md` or `RESEARCH_DESIGN.md` refer to internal design notes that
  are not part of this release.

## Licence

The code is released under the MIT licence (see `LICENSE`). The corpora and the base models
keep their own licences.
