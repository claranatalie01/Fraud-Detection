from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.modeling.metrics import CalibrationResult, best_f1_threshold, calibrate_platt, evaluate_binary_classifier
from src.modeling.xgb_runtime import resolve_xgb_compute
from src.preprocessing.baf_preprocessor import BAFPreprocessor, TimeSplit


def train_vanilla(
    data_path: str | Path,
    output_dir: str | Path = "results/vanilla",
    target_col: str = "fraud_bool",
    month_col: str = "month",
    prefer_gpu: bool = True,
    use_yeo_johnson: bool = True,
    use_smote: bool = True,
    smote_sampling_strategy: float = 0.5,
    smote_random_state: int = 42,
    fairness_group_cols: list[str] | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    split = TimeSplit()
    preprocessor = BAFPreprocessor(
        target_col=target_col,
        month_col=month_col,
        use_yeo_johnson=use_yeo_johnson,
    )
    train_df, valid_df, test_df = preprocessor.split_by_month(df, split)
    preprocessor.fit(train_df)
    preprocessor.save(output)

    X_train, y_train = preprocessor.transform_with_target(train_df)
    X_valid, y_valid = preprocessor.transform_with_target(valid_df)
    X_test, y_test = preprocessor.transform_with_target(test_df)
    X_train_fit, y_train_fit = X_train, y_train
    if use_smote:
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=smote_random_state)
        X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)

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
        X_train_fit,
        y_train_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    calibrator: CalibrationResult = calibrate_platt(y_valid.to_numpy(), valid_scores)
    test_scores_cal = calibrator.model.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    threshold = best_f1_threshold(y_valid.to_numpy(), valid_scores)
    fairness_groups = None
    if fairness_group_cols:
        present_cols = [c for c in fairness_group_cols if c in test_df.columns]
        fairness_groups = test_df[present_cols] if present_cols else None
    metrics = evaluate_binary_classifier(
        y_true=y_test.to_numpy(),
        y_score=test_scores,
        y_score_calibrated=test_scores_cal,
        threshold=threshold,
        groups=fairness_groups,
    )

    joblib.dump(model, output / "model.pkl")
    joblib.dump(calibrator.model, output / "platt_calibrator.pkl")
    pd.DataFrame({"score": test_scores, "label": y_test.to_numpy()}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame({"score_calibrated": test_scores_cal, "label": y_test.to_numpy()}).to_csv(output / "test_predictions_calibrated.csv", index=False)
    report = {
        "split": {"train_months": [0, 1, 2, 3, 4, 5], "valid_months": [6], "test_months": [7]},
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "metrics": metrics,
        "compute": compute,
        "preprocessing": {"use_yeo_johnson": use_yeo_johnson},
        "imbalance": {
            "method": "smote" if use_smote else "none",
            "sampling_strategy": smote_sampling_strategy if use_smote else None,
            "train_rows_before": int(len(X_train)),
            "train_rows_after": int(len(X_train_fit)),
        },
        "calibration": calibrator.__dict__,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate vanilla BAF XGBoost")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--output", default="results/vanilla")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    parser.add_argument("--disable-yeojohnson", action="store_true", help="Disable Yeo-Johnson transform")
    parser.add_argument("--disable-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.5)
    parser.add_argument("--smote-random-state", type=int, default=42)
    parser.add_argument("--fairness-group-cols", nargs="*", default=None)
    args = parser.parse_args()
    result = train_vanilla(
        args.data,
        args.output,
        prefer_gpu=not args.cpu_only,
        use_yeo_johnson=not args.disable_yeojohnson,
        use_smote=not args.disable_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
        smote_random_state=args.smote_random_state,
        fairness_group_cols=args.fairness_group_cols,
    )
    print(json.dumps(result, indent=2))
