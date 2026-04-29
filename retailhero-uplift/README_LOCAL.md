Local runtime notes
===================

This project was extracted from `retailhero-uplift.zip` and adjusted to run
locally with current Python tooling.

What changed
------------

- The top-level notebook [Uplift_Modeling_Phase_1.ipynb](</D:/Business Analytics/Uplift_Modeling_Phase_1.ipynb>)
  now resolves the extracted `retailhero-uplift` folder automatically.
- Heavy purchase aggregations were moved into `retailhero_features.py`, which
  uses DuckDB to query the 4.4 GB CSV efficiently without loading it fully
  into memory.
- `uplift_solution.py` now uses local paths and writes `outputs/submission.csv`.

Setup
-----

Run:

```powershell
python -m pip install -r .\requirements.txt
```

Notebook outputs
----------------

The notebook writes:

- `outputs/master_train.csv`
- `outputs/master_test.csv`

Starter model output
--------------------

Run:

```powershell
python .\uplift_solution.py
```

This writes:

- `outputs/submission.csv`

Phase II pipeline
-----------------

Run:

```powershell
python .\phase2_pipeline.py --reuse-features
```

This writes:

- `outputs/phase2/phase2_scored_validation.csv`
- `outputs/phase2/phase2_scored_test.csv`
- `outputs/phase2/phase2_summary_metrics.csv`
- `outputs/phase2/phase2_qini_curves.csv`
- `outputs/phase2/phase2_liftup_curves.csv`
- `outputs/phase2/phase2_profit_sensitivity.csv`
- `outputs/phase2/qini_curves.png`
- `outputs/phase2/liftup_curves.png`
- `outputs/phase2/profit_vs_target_depth.png`
- `outputs/phase2/profit_sensitivity.png`

Verify the Phase II outputs with:

```powershell
python .\verify_phase2_outputs.py
```
