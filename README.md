# BAF Agentic Fraud Detection

This repository implements an account-level fraud pipeline with:
- leakage-safe BAF preprocessing and time-based split,
- vanilla XGBoost baseline model,
- retriever-enriched XGBoost model,
- statistical comparison and model selection,
- real-time inference with SHAP and LLM-generated fraud report.

## Time split protocol

- Train: months `0-5`
- Validation: month `6`
- Test: month `7`

This split is enforced before fitting preprocessing artifacts to avoid leakage.

## Repository layout

- `src/preprocessing/baf_preprocessor.py`: BAF-only deterministic preprocessing.
- `src/modeling/train_vanilla.py`: train/evaluate vanilla model.
- `src/modeling/train_enriched.py`: train/evaluate retriever-enriched model.
- `src/modeling/compare_models.py`: bootstrap delta PR-AUC and model selection.
- `src/retriever/A2A.py`: migrated retriever backend.
- `src/retriever/enrichment.py`: convert retriever output into model features.
- `src/inference/account_inference.py`: real-time score + SHAP + report payload.
- `src/inference/app.py`: Flask endpoint for account fraud report generation.
- `scripts/run_experiment.py`: end-to-end experiment runner.

## Setup

```bash
pip install -r requirements.txt
```

### GPU training (RTX 4090 / CUDA 12.5 / driver 560+)

- The training scripts auto-detect CUDA and use GPU XGBoost when available.
- Runtime settings used for GPU:
  - `tree_method = hist`
  - `device = cuda`
- If CUDA is unavailable, scripts automatically fall back to CPU.
- To force CPU explicitly:
  - `python scripts/run_experiment.py --data /path/to/base.csv --cpu-only`

If you use LLM report generation, set:
- `DEEPSEEK_API_KEY`
- optional `DEEPSEEK_BASE_URL`
- optional `DEEPSEEK_MODEL`

If you use retriever enrichment, run the retriever PostgreSQL/pgvector backend first.

## Run experiment

```bash
python scripts/run_experiment.py --data /path/to/base.csv --results-dir results
```

Outputs:
- `results/vanilla/metrics.json`
- `results/enriched/metrics.json`
- `results/model_selection.json`
- `results/summary.json`

## Real-time account inference

Start API:

```bash
python -m src.inference.app
```

POST payload to `/agent/account_fraud_report`:

```json
{
  "id": "request-001",
  "input": {
    "query": {
      "month": 6,
      "income": 0.8
    }
  }
}
```

Response contains:
- `fraud_score`
- `recommended_action` (`approve | escalate | reject`)
- top SHAP drivers
- retriever evidence summary
- LLM narrative report text

## Notes

- Current scope is account-level only.
- NVIDIA transaction model notebooks are retained only as references.
- Retriever enrichment must preserve temporal constraints (`neighbor_month < query_month`).