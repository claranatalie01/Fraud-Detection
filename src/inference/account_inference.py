from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from openai import OpenAI

from src.preprocessing.baf_preprocessor import BAFPreprocessor
from src.retriever.enrichment import build_retriever_features_for_records


@dataclass(frozen=True)
class DecisionPolicy:
    approve_threshold: float = 0.30
    escalate_threshold: float = 0.65


def recommend_action(score: float, policy: DecisionPolicy) -> str:
    if score < policy.approve_threshold:
        return "approve"
    if score < policy.escalate_threshold:
        return "escalate"
    return "reject"


def _top_shap_features(explainer: shap.Explainer | None, model: Any, X: pd.DataFrame, top_n: int = 5) -> list[dict[str, float]]:
    if explainer is not None:
        values = explainer(X)
        shap_values = values.values[0]
    else:
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = contrib[0, :-1]
    ranked = np.argsort(np.abs(shap_values))[::-1][:top_n]
    return [{"feature": X.columns[i], "shap_value": float(shap_values[i])} for i in ranked]


def _generate_llm_report(payload: dict[str, Any]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return "LLM report unavailable: DEEPSEEK_API_KEY not set."

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        "Write a concise fraud analyst report using only provided facts. "
        "Include score interpretation, top drivers, retriever evidence, and final recommendation."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fraud risk analyst. Do not invent facts."},
            {"role": "user", "content": prompt + "\n\nFacts:\n" + json.dumps(payload, indent=2)},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


class AccountFraudInferenceService:
    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
        calibrator_path: str | Path | None = None,
        enriched: bool = True,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.model = joblib.load(model_path)
        self.preprocessor: BAFPreprocessor = BAFPreprocessor.load(preprocessor_path)
        self.calibrator = joblib.load(calibrator_path) if calibrator_path else None
        self.enriched = enriched
        self.policy = policy or DecisionPolicy()
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    def score(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_df = pd.DataFrame([record])
        X = self.preprocessor.transform_records(raw_df)
        retriever_features = {}
        if self.enriched:
            retr = build_retriever_features_for_records(raw_df)
            retriever_features = retr.iloc[0].to_dict()
            X = pd.concat([X, retr], axis=1)

        score = float(self.model.predict_proba(X)[:, 1][0])
        if self.calibrator is not None:
            score = float(self.calibrator.predict_proba(np.array([[score]], dtype=float))[:, 1][0])
        action = recommend_action(score, self.policy)
        top_drivers = _top_shap_features(self.explainer, self.model, X, top_n=5)
        grounded = {
            "fraud_score": score,
            "decision_policy": asdict(self.policy),
            "recommended_action": action,
            "top_shap_drivers": top_drivers,
            "retriever_evidence": retriever_features,
        }
        report_text = _generate_llm_report(grounded)
        return {
            "report": grounded,
            "llm_report_text": report_text,
        }

    @classmethod
    def from_champion_manifest(
        cls,
        manifest_path: str | Path,
        variant_name: str,
        enriched: bool = True,
        policy: DecisionPolicy | None = None,
    ) -> "AccountFraudInferenceService":
        payload = json.loads(Path(manifest_path).read_text())
        champion = payload["overall_champion"]
        model_path = payload["artifact_template"].format(variant=variant_name, champion=champion)
        calibrator_path = payload["calibrator_template"].format(variant=variant_name, champion=champion)
        preprocessor_path = payload["preprocessor_template"].format(variant=variant_name, champion=champion)
        return cls(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            calibrator_path=calibrator_path,
            enriched=enriched,
            policy=policy,
        )
