# RetailHero Uplift Modeling Pipeline

This repository contains the working codebase for a two-phase uplift-modeling project built on the RetailHero / X5 retail dataset.

The repo is intentionally code-first and reproducible:
- Phase I established and verified the empirical findings that justify uplift modeling.
- Phase II implemented a comparable `CCP` vs `CCU` modeling pipeline with validation, ranking outputs, and profit-oriented evaluation.

The repository does **not** include the raw data or generated outputs. Those are excluded on purpose because the purchase history file is large and because model outputs should be reproducible from code.

## Repository Structure

- `Uplift_Modeling_Phase_1.ipynb`
  - Phase I exploratory notebook.
  - Local-path patched so it can run outside Colab.
- `retailhero-uplift/`
  - Main project folder.
- `retailhero-uplift/retailhero_features.py`
  - Shared DuckDB-backed helpers for large-CSV access and purchase aggregation.
- `retailhero-uplift/phase2_features.py`
  - Phase II feature engineering from raw CSVs.
- `retailhero-uplift/phase2_models.py`
  - Model wrappers for CCP and CCU estimators.
- `retailhero-uplift/phase2_metrics.py`
  - Qini, liftup, MP, and MPU evaluation logic.
- `retailhero-uplift/phase2_pipeline.py`
  - End-to-end Phase II entrypoint.
- `retailhero-uplift/verify_phase1_findings.py`
  - Recomputes Phase I claims directly from the raw data.
- `retailhero-uplift/verify_phase2_outputs.py`
  - Validates the saved Phase II artifacts.
- `retailhero-uplift/README_LOCAL.md`
  - Local run notes focused on this machine setup.

## Data Expectations

The code expects the extracted competition files under:

```text
retailhero-uplift/data/
```

Expected files:
- `clients.csv`
- `products.csv`
- `purchases.csv`
- `uplift_train.csv`
- `uplift_test.csv`
- `uplift_sample_submission.csv`

The repo excludes:
- `retailhero-uplift/data/`
- `retailhero-uplift/outputs/`
- `retailhero-uplift.zip`

That is deliberate. Raw data is large and should be shared separately through secure storage or an internal shared drive.

## Setup

Use Python 3.13 on Windows, then install:

```powershell
cd "D:\Business Analytics\retailhero-uplift"
python -m pip install -r .\requirements.txt
```

Core dependencies:
- `duckdb` for large CSV aggregation without loading the full 4.4 GB purchases file into memory
- `scikit-learn` for logistic uplift models
- `xgboost` for the CCP baseline
- `causalml` for uplift random forest
- `matplotlib` / `seaborn` for plots

## How To Run

### Phase I verification

```powershell
cd "D:\Business Analytics\retailhero-uplift"
python .\verify_phase1_findings.py
```

This recomputes the Phase I findings from raw data instead of trusting slide text or notebook prose.

### Phase II pipeline

```powershell
cd "D:\Business Analytics\retailhero-uplift"
python .\phase2_pipeline.py --reuse-features
```

Optional flags:
- `--data-dir`
- `--output-dir`
- `--random-seed`
- `--sms-cost`
- `--avg-basket-value`
- `--skip-sensitivity-sweep`

### Phase II output verification

```powershell
cd "D:\Business Analytics\retailhero-uplift"
python .\verify_phase2_outputs.py
```

## What Phase I Established

Phase I was not just descriptive EDA. It established the causal and behavioral basis for Phase II.

Verified findings:
- The treatment/control split is effectively randomized.
- The baseline treatment effect is modest at roughly `3.32 pp`.
- Uplift is strongest among lower-frequency shoppers.
- Around `30.2%` of clients are proxy `Sure Things`, which makes a pure purchase-propensity strategy wasteful.
- `age` contains corrupt values and must be cleaned explicitly.
- The product taxonomy is anonymized, so meaningful product features must come from flags and behavioral summaries rather than category names.
- All test-set clients have purchase history, so the Phase II feature matrix can be complete without special-case imputation for missing transaction history.

These findings matter because they justify why Phase II should compare a standard response model against uplift-specific models instead of only optimizing raw purchase probability.

## What Phase II Implements

Phase II implements a CCP vs CCU comparison under one reproducible pipeline.

### Model A: CCP baseline

`CCP` here means a standard customer purchase-propensity ranking.

Implementation:
- `XGBClassifier`
- trained on the **control group only**
- predicts `P(buy)` for all customers
- evaluated with a predictive maximum-profit framing

Why this is defensible:
- It matches the roadmap requirement.
- It creates a strong conventional benchmark.
- Training on control only avoids contaminating the baseline with treatment-induced outcomes when the goal is to estimate organic purchase propensity.

### Model B1: Two-model logistic uplift

Implementation:
- one logistic model on treated customers
- one logistic model on control customers
- uplift score = `P(buy | treated) - P(buy | control)`

Why this is defensible:
- It is the most literal implementation of the roadmap’s uplift formula.
- It is simple, transparent, and easy to diagnose.
- It provides an interpretable bridge between classical churn/response modeling and causal uplift ranking.

### Model B2: Lo-style interaction logistic uplift

Implementation:
- one logistic model
- treatment indicator included as a feature
- first-order treatment interactions with every feature
- uplift score computed by scoring each customer twice: once with `t=1`, once with `t=0`

Why this is defensible:
- It operationalizes the “single model with interactions” family referenced in the roadmap.
- It enforces a shared parameterization across treatment and control instead of fitting two unrelated surfaces.
- It provides a useful contrast to the two-model approach.

### Model B3: Uplift random forest

Implementation:
- `causalml.inference.tree.UpliftRandomForestClassifier`
- binary treatment recoded to the API’s required string labels
- `KL` splitting criterion
- light hyperparameter tuning on tree count, depth, leaf size, and regularization

Why this is defensible:
- It is the roadmap’s non-linear uplift model.
- It can capture heterogeneous treatment effects and non-linear feature interactions that logistic models cannot.
- In the current validation run, it is the best CCU model by both `Qini` and `MPU`.

## Feature Engineering Rationale

The Phase II feature matrix is built directly from raw CSVs, not from notebook-only exports.

Feature groups:
- cleaned demographics
  - `age_clean`, `age_flagged`, gender indicators, `gender_known`
- redemption behavior
  - `ever_redeemed`, `days_to_redeem`, issue-age and redeem-recency features
- purchase behavior
  - total spend, transactions, trips, items, basket stats, stores, products, points, recency, tenure
- product behavior
  - alcohol spend share
  - own-trademark spend share
  - netto summaries
  - spend-share across the 3 hashed `level_1` groups
  - brand concentration (`HHI`)

Why this is defensible:
- It follows directly from the verified Phase I findings.
- It avoids pretending that anonymized category labels are interpretable.
- It converts product metadata into behaviorally meaningful client-level signals.
- It creates a fully numeric matrix with explicit missingness indicators, which keeps model behavior deterministic and comparable across estimators.

## Evaluation Rationale

The project intentionally does **not** stop at AUC-style classification scoring because that would not answer the targeting question.

Implemented evaluation layers:
- `Qini` curves
- `liftup` curves
- CCP predictive maximum profit (`MP`)
- CCU maximum profit uplift (`MPU`)
- sensitivity analysis over basket value and SMS cost

Why this is defensible:
- Uplift modeling is about **incremental value**, not just purchase likelihood.
- A customer who is very likely to buy anyway is not necessarily a good marketing target.
- Profit-oriented ranking is the correct business-facing evaluation when the intervention has a cost.

Default economics:
- average basket value = empirical average trip value from the purchase history (`~4425 RUB`)
- SMS cost = `1 RUB`

Those defaults are configurable in the CLI and the pipeline also produces sensitivity outputs so the conclusions are not tied to one brittle assumption.

## Current Phase II Result Snapshot

From the current full validation run:

- `Model B3 - CCU Uplift Random Forest`
  - `qini_auc = 0.004759`
  - `mpu = 158.307694`
  - `mpu_optimal_target_share = 0.7475`
- `Model A - CCP Baseline`
  - `predictive_mp = 2742.063982`
  - `mpu = 146.083912`
- `Model B1 - CCU Two-Model Logistic`
  - `qini_auc = 0.003297`
  - `mpu = 146.083912`
- `Model B2 - CCU Lo Interaction Logistic`
  - `qini_auc = 0.003245`
  - `mpu = 146.083912`

Interpretation:
- The uplift random forest is currently the strongest CCU model.
- The logistic uplift models do show positive uplift discrimination.
- The CCP baseline is still useful as a benchmark, but it is not the best model for incremental targeting.

## Output Files

The pipeline writes into:

```text
retailhero-uplift/outputs/phase2/
```

Key artifacts:
- `phase2_scored_validation.csv`
- `phase2_scored_test.csv`
- `phase2_summary_metrics.csv`
- `phase2_qini_curves.csv`
- `phase2_liftup_curves.csv`
- `phase2_profit_sensitivity.csv`
- `qini_curves.png`
- `liftup_curves.png`
- `profit_vs_target_depth.png`
- `profit_sensitivity.png`

## How Collaborators Should Work

Recommended workflow:
- clone the repo
- obtain the raw data separately
- place the files under `retailhero-uplift/data/`
- install dependencies
- run `verify_phase1_findings.py`
- run `phase2_pipeline.py --reuse-features`
- run `verify_phase2_outputs.py`

When making changes:
- preserve the raw-data-first workflow
- do not commit `data/` or `outputs/`
- prefer adding new metrics or model variants as new functions/modules rather than overloading the existing pipeline with hidden branches

## Limitations

Important limitations to understand:
- The hashed category hierarchy limits semantic product interpretation.
- The profit analysis is only as good as the retail economics assumptions.
- `causalml` uplift forests are heavier and slower than the logistic baselines.
- The current implementation is a strong experimental pipeline, not a production service.

Those are acceptable constraints for this stage of the project because the current objective is methodological comparison and defensible analysis, not deployment.

## Bottom Line

This repository is structured to answer one question cleanly:

> Is a value-aware uplift targeting strategy better justified than a standard purchase-propensity ranking on this retail intervention dataset?

The code, validation scripts, and outputs were built to make that argument reproducible, inspectable, and easy for collaborators to extend.
