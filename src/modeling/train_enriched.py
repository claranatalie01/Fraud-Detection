from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.modeling.metrics import best_f1_threshold, evaluate_binary_classifier
from src.modeling.xgb_runtime import resolve_xgb_compute
from src.preprocessing.baf_preprocessor import BAFPreprocessor, TimeSplit
from src.retriever.enrichment import build_retriever_features_for_records


def train_enriched(
    data_path: str | Path,
    output_dir: str | Path = "results/enriched",
    target_col: str = "fraud_bool",
    month_col: str = "month",
    prefer_gpu: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    split = TimeSplit()
    preprocessor = BAFPreprocessor(target_col=target_col, month_col=month_col)
    train_df, valid_df, test_df = preprocessor.split_by_month(df, split)
    preprocessor.fit(train_df)
    preprocessor.save(output)

    X_train, y_train = preprocessor.transform_with_target(train_df)
    X_valid, y_valid = preprocessor.transform_with_target(valid_df)
    X_test, y_test = preprocessor.transform_with_target(test_df)

    retr_train = build_retriever_features_for_records(train_df.drop(columns=[target_col]))
    retr_valid = build_retriever_features_for_records(valid_df.drop(columns=[target_col]))
    retr_test = build_retriever_features_for_records(test_df.drop(columns=[target_col]))

    X_train_enriched = pd.concat([X_train, retr_train], axis=1)
    X_valid_enriched = pd.concat([X_valid, retr_valid], axis=1)
    X_test_enriched = pd.concat([X_test, retr_test], axis=1)

    compute = resolve_xgb_compute(prefer_gpu=prefer_gpu)
    model = XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        tree_method=compute["tree_method"],
        device=compute["device"],
    )
    model.fit(
        X_train_enriched,
        y_train,
        eval_set=[(X_valid_enriched, y_valid)],
        verbose=False,
    )

    valid_scores = model.predict_proba(X_valid_enriched)[:, 1]
    test_scores = model.predict_proba(X_test_enriched)[:, 1]
    threshold = best_f1_threshold(y_valid.to_numpy(), valid_scores)
    metrics = evaluate_binary_classifier(y_test.to_numpy(), test_scores, threshold)

    joblib.dump(model, output / "model.pkl")
    pd.DataFrame({"score": test_scores, "label": y_test.to_numpy()}).to_csv(output / "test_predictions.csv", index=False)
    report = {
        "split": {"train_months": [0, 1, 2, 3, 4, 5], "valid_months": [6], "test_months": [7]},
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "metrics": metrics.__dict__,
        "retriever_features": retr_train.columns.tolist(),
        "compute": compute,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate enriched BAF XGBoost")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--output", default="results/enriched")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    args = parser.parse_args()
    result = train_enriched(args.data, args.output, prefer_gpu=not args.cpu_only)
    print(json.dumps(result, indent=2))
