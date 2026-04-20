"""
writer_agent.py - Updated with real SHAP values
"""

import os
from flask import Flask, request, jsonify
from openai import OpenAI
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

app = Flask(__name__)

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
PRIMARY_MODEL = "deepseek-chat"
FALLBACK_MODEL = "deepseek-reasoner"

# Initialize Risk Scoring Agent
from risk_scoring_agent import RiskScoringAgent
risk_agent = RiskScoringAgent()

def compute_feature_stats(similar_cases, feature_name):
    """Compute statistical summary for a numerical feature."""
    fraud_vals = []
    legit_vals = []
    for case in similar_cases:
        val = case.get('metadata', {}).get(feature_name)
        if val is None or not isinstance(val, (int, float)):
            continue
        if case['fraud_bool'] == 1:
            fraud_vals.append(val)
        else:
            legit_vals.append(val)
    
    stats = {}
    if fraud_vals:
        stats['fraud_min'] = min(fraud_vals)
        stats['fraud_max'] = max(fraud_vals)
        stats['fraud_mean'] = sum(fraud_vals) / len(fraud_vals)
        stats['fraud_count'] = len(fraud_vals)
    if legit_vals:
        stats['legit_min'] = min(legit_vals)
        stats['legit_max'] = max(legit_vals)
        stats['legit_mean'] = sum(legit_vals) / len(legit_vals)
        stats['legit_count'] = len(legit_vals)
    
    if fraud_vals and legit_vals:
        if stats['fraud_min'] > stats['legit_max']:
            stats['separation'] = True
            stats['threshold'] = (stats['legit_max'] + stats['fraud_min']) / 2
        elif stats['fraud_max'] < stats['legit_min']:
            stats['separation'] = True
            stats['threshold'] = (stats['fraud_max'] + stats['legit_min']) / 2
        else:
            stats['separation'] = False
    else:
        stats['separation'] = False
    return stats

def get_best_cases(similar_cases):
    """Get most similar fraudulent and legitimate cases."""
    best_fraud = None
    best_legit = None
    for case in similar_cases:
        if case['fraud_bool'] == 1:
            if best_fraud is None or case['similarity'] > best_fraud['similarity']:
                best_fraud = case
        else:
            if best_legit is None or case['similarity'] > best_legit['similarity']:
                best_legit = case
    return best_fraud, best_legit

def build_prompt(query_metadata, similar_cases, local_fraud_rate, risk_assessment, question):
    """Build prompt with real SHAP values and separate contextual indicators."""
    
    # ============================================================
    # SECTION 1: SHAP-based indicators (from the application's own features)
    # ============================================================
    shap_features = risk_assessment.get('top_shap_features', [])
    shap_text = ""
    for feat in shap_features[:5]:
        direction = "INCREASES risk" if feat['shap_value'] > 0 else "DECREASES risk"
        shap_text += f"- **{feat['feature']}**: {direction} by {abs(feat['shap_value']):.3f}\n"
    
    if not shap_text:
        shap_text = "No SHAP values available."
    
    # ============================================================
    # SECTION 2: Contextual indicators (from similar past cases)
    # ============================================================
    best_fraud, best_legit = get_best_cases(similar_cases)
    
    contextual_text = ""
    if best_fraud:
        contextual_text += f"- **{best_fraud['similarity']:.0%} similarity to known fraud case** (ID: {best_fraud['id']})\n"
        
        # Extract risk patterns from the similar fraud case
        fraud_meta = best_fraud['metadata']
        if fraud_meta.get('velocity_6h', 0) > 10000:
            contextual_text += f"- High transaction velocity ({fraud_meta['velocity_6h']:.0f} in 6h) in similar fraud case\n"
        if fraud_meta.get('prev_address_months_count', 0) < 6:
            contextual_text += f"- Anomalous address history in similar fraud case\n"
        if fraud_meta.get('name_email_similarity', 1) < 0.3:
            contextual_text += f"- Name-email mismatch in similar fraud case\n"
    
    if not contextual_text and best_fraud:
        contextual_text = f"- {best_fraud['similarity']:.0%} similar to a known fraudulent case\n"
    
    # ============================================================
    # SECTION 3: Query features (full list)
    # ============================================================
    query_summary = {}
    numerical_features = [
        'income', 'name_email_similarity', 'prev_address_months_count',
        'current_address_months_count', 'customer_age', 'days_since_request',
        'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
        'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
        'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',
        'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count',
        'month'
    ]
    categorical_features = [
        'payment_type', 'employment_status', 'email_is_free', 'housing_status',
        'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request',
        'source', 'device_os', 'keep_alive_session'
    ]
    
    for feat in numerical_features + categorical_features:
        query_summary[feat] = query_metadata.get(feat, '?')
    
    # ============================================================
    # SECTION 4: Statistical analysis (numerical feature patterns)
    # ============================================================
    stats_text = ""
    for feat in numerical_features:
        stats = compute_feature_stats(similar_cases, feat)
        if not stats:
            continue
        if stats.get('separation'):
            stats_text += f"- **{feat}**: Clear separation. Fraudulent cases have {feat} ≥ {stats['fraud_min']:.1f}, legitimate cases have {feat} ≤ {stats['legit_max']:.1f}.\n"
        else:
            if 'fraud_mean' in stats and 'legit_mean' in stats:
                ratio = stats['fraud_mean'] / (stats['legit_mean'] + 0.001)
                if ratio > 2 or ratio < 0.5:
                    stats_text += f"- **{feat}**: Notable difference – fraudulent average {stats['fraud_mean']:.1f}, legitimate average {stats['legit_mean']:.1f}.\n"
    
    if not stats_text:
        stats_text = "No strong numerical signals detected."
    
    # ============================================================
    # SECTION 5: Best similar cases details
    # ============================================================
    best_fraud_text = ""
    best_legit_text = ""
    if best_fraud:
        meta = best_fraud['metadata']
        best_fraud_text = f"ID={best_fraud['id']}, similarity={best_fraud['similarity']:.3f}\n"
        for feat in numerical_features[:5]:
            best_fraud_text += f"  {feat}: {meta.get(feat, '?')}\n"
    if best_legit:
        meta = best_legit['metadata']
        best_legit_text = f"ID={best_legit['id']}, similarity={best_legit['similarity']:.3f}\n"
        for feat in numerical_features[:5]:
            best_legit_text += f"  {feat}: {meta.get(feat, '?')}\n"
    
    # ============================================================
    # SECTION 6: Task based on question
    # ============================================================
    if question and question.strip():
        task = f"Answer the following question using ONLY the data above. Question: {question}"
        extra_instructions = "Answer ONLY the question. Do not add default sections."
    else:
        task = """Provide a fraud risk assessment with this structure:

1. **Verdict** – Final decision (APPROVE/ESCALATE/REJECT) and confidence
2. **Risk Score Breakdown** – ML score, local fraud rate, final score, weights used
3. **SHAP Analysis** – Top 5 features influencing this decision
4. **Contextual Risk Indicators** – Similarity to known fraud cases and patterns
5. **Key Risk Indicators** – 3-5 specific concerns (mix of SHAP and contextual)
6. **Recommendation** – Clear next steps"""
        extra_instructions = "Follow the exact structure above. Separate feature-based risks from contextual risks."
    
    # ============================================================
    # SECTION 7: Assemble final prompt
    # ============================================================
    prompt = f"""
You are a professional bank fraud investigator.

**QUERY APPLICATION (key features):**
{query_summary}

**RISK ASSESSMENT SUMMARY:**
- ML Model Score: {risk_assessment.get('ml_score', 0):.3f}
- Local Fraud Rate: {risk_assessment.get('local_fraud_rate', 0):.3f}
- Final Risk Score: {risk_assessment.get('final_score', 0):.3f}
- Recommendation: {risk_assessment.get('recommendation', 'UNKNOWN')}
- Weights: ML={risk_assessment.get('weights_used', {}).get('ml_score', 0.6)}, Retriever={risk_assessment.get('weights_used', {}).get('local_fraud_rate', 0.4)}

**SECTION 1: FEATURE-BASED RISK INDICATORS (from SHAP analysis)**
These are based on the application's own features:
{shap_text}

**SECTION 2: CONTEXTUAL RISK INDICATORS (from similar past cases)**
{contextual_text if contextual_text else "No similar fraud cases found"}

**SECTION 3: STATISTICAL SIGNALS (from similar cases)**
{stats_text}

**SECTION 4: MOST SIMILAR CASES**
- Local fraud rate among {risk_assessment.get('similar_cases_count', 0)} similar cases: {local_fraud_rate:.2%}

**Most similar fraudulent case:**
{best_fraud_text if best_fraud_text else "None"}

**Most similar legitimate case:**
{best_legit_text if best_legit_text else "None"}

**TASK:**
{task}

**INSTRUCTIONS:**
- Base your answer ONLY on the provided data.
- Be factual and concise.
- Clearly distinguish between feature-based risks (SHAP) and contextual risks (similar cases).
- {extra_instructions}
"""
    return prompt

@app.route('/write', methods=['POST'])
def write():
    data = request.json
    query_metadata = data.get('query_metadata', {})
    similar_cases = data.get('similar_cases', [])
    local_fraud_rate = data.get('local_fraud_rate', 0)
    risk_assessment = data.get('risk_assessment', {})  # ← Changed from risk_score
    question = data.get('question', None)
    
    if not similar_cases:
        return jsonify({"narrative": "No similar cases found. Cannot generate assessment."})
    
    # Get risk assessment with real SHAP values
    risk_assessment = risk_agent.assess_application(query_metadata, {
        'local_fraud_rate': local_fraud_rate,
        'similar_cases': similar_cases,
        'total_neighbors': len(similar_cases)
    })
    
    prompt = build_prompt(query_metadata, similar_cases, local_fraud_rate, risk_assessment, question)
    
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    narrative = None
    last_error = None
    
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional bank fraud investigator. Always be concise and factual."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            if hasattr(response, 'choices') and response.choices:
                narrative = response.choices[0].message.content
                break
            else:
                last_error = "API response missing 'choices'"
                continue
        except Exception as e:
            last_error = str(e)
            continue
    
    if narrative is None:
        narrative = f"Error generating narrative: {last_error}"
    
    # Include risk assessment in response
    return jsonify({
        "narrative": narrative,
        "risk_assessment": risk_assessment
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)