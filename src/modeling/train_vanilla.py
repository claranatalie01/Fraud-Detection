from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.modeling.metrics import best_f1_threshold, evaluate_binary_classifier
from src.modeling.xgb_runtime import resolve_xgb_compute
from src.preprocessing.baf_preprocessor import BAFPreprocessor, TimeSplit


def train_vanilla(
    data_path: str | Path,
    output_dir: str | Path = "results/vanilla",
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

    compute = resolve_xgb_compute(prefer_gpu=prefer_gpu)
    model = XGBClassifier(
        n_estimators=400,
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
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold = best_f1_threshold(y_valid.to_numpy(), valid_scores)
    metrics = evaluate_binary_classifier(y_test.to_numpy(), test_scores, threshold)

    joblib.dump(model, output / "model.pkl")
    pd.DataFrame({"score": test_scores, "label": y_test.to_numpy()}).to_csv(output / "test_predictions.csv", index=False)
    report = {
        "split": {"train_months": [0, 1, 2, 3, 4, 5], "valid_months": [6], "test_months": [7]},
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "metrics": metrics.__dict__,
        "compute": compute,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate vanilla BAF XGBoost")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--output", default="results/vanilla")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    args = parser.parse_args()
    result = train_vanilla(args.data, args.output, prefer_gpu=not args.cpu_only)
    print(json.dumps(result, indent=2))
