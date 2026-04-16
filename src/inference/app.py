from __future__ import annotations

from flask import Flask, jsonify, request

from src.inference.account_inference import AccountFraudInferenceService


def create_app(
    model_path: str = "results/enriched/model.pkl",
    preprocessor_path: str = "results/enriched/baf_preprocessor.pkl",
    enriched: bool = True,
) -> Flask:
    app = Flask(__name__)
    service = AccountFraudInferenceService(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        enriched=enriched,
    )

    @app.post("/agent/account_fraud_report")
    def account_fraud_report():
        task = request.get_json(silent=True) or {}
        record = task.get("input", {}).get("query")
        if not record:
            return jsonify({"error": "Missing input.query payload"}), 400
        try:
            output = service.score(record)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"task_id": task.get("id", "unknown"), "output": output})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5002, debug=True)
