import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from risk_scoring_agent import RiskScoringAgent

# Configuration
RETRIEVER_URL = "http://localhost:5001/retrieve"
WRITER_URL = "http://localhost:5004/write"

st.set_page_config(page_title="Fraud Detection Assistant", layout="wide")
st.title("🕵️ Fraud Detection Assistant")

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
if "retriever_data" not in st.session_state:
    st.session_state.retriever_data = None          # cached retriever output
if "risk_assessment" not in st.session_state:
    st.session_state.risk_assessment = None         # cached risk assessment
if "current_app_id" not in st.session_state:
    st.session_state.current_app_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []              # list of {"role": "user"/"assistant", "content": ...}
if "last_narrative" not in st.session_state:
    st.session_state.last_narrative = ""

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def fetch_retriever_data(app_id):
    """Call retriever agent and return parsed JSON or None."""
    try:
        resp = requests.post(RETRIEVER_URL, json={"application_id": int(app_id)}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            st.error(f"Retriever error: {data['error']}")
            return None
        return data
    except Exception as e:
        st.error(f"Error calling Retriever Agent: {e}")
        return None

def call_risk_scoring(query_metadata, retriever_output):
    """Call risk scoring to get combined score with SHAP."""
    agent = RiskScoringAgent()
    return agent.assess_application(query_metadata, retriever_output)

def call_writer_agent(query_metadata, similar_cases, local_fraud_rate, risk_assessment, question):
    """Call writer agent and return narrative string."""
    payload = {
        "query_metadata": query_metadata,
        "similar_cases": similar_cases,
        "local_fraud_rate": local_fraud_rate,
        "risk_assessment": risk_assessment,  # Pass full risk assessment
        "question": question if question and question.strip() else None
    }
    try:
        resp = requests.post(WRITER_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json().get("narrative", "No narrative generated.")
    except Exception as e:
        return f"Error generating narrative: {e}"

def display_similar_cases_table(similar_cases):
    """Display similar cases table."""
    all_features = ['income', 'velocity_6h', 'payment_type', 'prev_address_months_count', 
                    'intended_balcon_amount', 'name_email_similarity', 'zip_count_4w',
                    'credit_risk_score', 'customer_age']
    selected_features = st.multiselect("Show features in table", all_features, default=all_features[:3], key="table_features")
    
    if similar_cases:
        df_data = []
        for case in similar_cases:
            meta = case.get("metadata", {})
            row = {
                "ID": case["id"],
                "Fraud": "⚠️ Yes" if case["fraud_bool"] else "✅ No",
                "Month": case["month"],
                "Similarity": f"{case['similarity']:.3f}"
            }
            for feat in selected_features:
                row[feat] = meta.get(feat, "?")
            df_data.append(row)
        df = pd.DataFrame(df_data)
        st.subheader("📋 Most Similar Past Applications")
        st.dataframe(df, use_container_width=True, height=400)
        
        with st.expander("🔍 View Full Metadata of a Similar Case"):
            case_id = st.selectbox("Select Case ID", [c["id"] for c in similar_cases], key="meta_select")
            selected_meta = next(c["metadata"] for c in similar_cases if c["id"] == case_id)
            st.json(selected_meta)
    else:
        st.warning("No similar cases found.")

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Analysis Controls")
    app_id = st.number_input("Application ID", min_value=1, step=1, value=856982, key="app_id_input")
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("🔄 New Analysis", use_container_width=True):
        # Fetch new data
        with st.spinner("Fetching similar cases..."):
            new_data = fetch_retriever_data(app_id)
        if new_data:
            st.session_state.retriever_data = new_data
            st.session_state.current_app_id = app_id
            st.session_state.chat_history = []
            st.session_state.last_narrative = ""
            
            # Get risk assessment with SHAP
            with st.spinner("Calculating risk score with SHAP..."):
                risk_assessment = call_risk_scoring(
                    new_data["query_metadata"],
                    {
                        "local_fraud_rate": new_data["local_fraud_rate"],
                        "similar_cases": new_data["similar_cases"],
                        "total_neighbors": new_data["total_neighbors"]
                    }
                )
                st.session_state.risk_assessment = risk_assessment
            
            # Generate default report (empty question → full report)
            with st.spinner("Generating initial report..."):
                narrative = call_writer_agent(
                    new_data["query_metadata"],
                    new_data["similar_cases"],
                    new_data["local_fraud_rate"],
                    risk_assessment,
                    None
                )
                st.session_state.last_narrative = narrative
                st.session_state.chat_history.append({"role": "assistant", "content": narrative})
            st.rerun()
    
    if col_btn2.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.retriever_data = None
        st.session_state.risk_assessment = None
        st.session_state.current_app_id = None
        st.session_state.chat_history = []
        st.session_state.last_narrative = ""
        st.rerun()
    
    st.divider()
    
    # Display weights information
    st.caption("**Risk Scoring Weights**")
    st.caption("- ML Model: 60%")
    st.caption("- Local Fraud Rate: 40%")
    
    st.divider()
    st.caption("**Instructions**\n- Enter an Application ID and click 'New Analysis'.\n- Ask follow‑up questions in the chat box below.\n- Each follow‑up uses the same cached data (no refetch).\n- To analyse a different ID, click 'New Analysis' again.")

# ------------------------------------------------------------------
# Main area: chat interface and analysis display
# ------------------------------------------------------------------
if st.session_state.retriever_data is not None and st.session_state.risk_assessment is not None:
    ret_data = st.session_state.retriever_data
    risk_assessment = st.session_state.risk_assessment
    
    # ------------------------------------------------------------------
    # Risk Metrics Display
    # ------------------------------------------------------------------
    st.subheader("📊 Risk Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "ML Model Score", 
            f"{risk_assessment['ml_score']:.2%}",
            help="XGBoost model fraud probability based on application patterns"
        )
    
    with col2:
        st.metric(
            "Local Fraud Rate", 
            f"{risk_assessment['local_fraud_rate']:.2%}",
            help=f"Fraud rate among {risk_assessment.get('similar_cases_count', 0)} similar past applications"
        )
    
    with col3:
        st.metric(
            "Final Risk Score", 
            f"{risk_assessment['final_score']:.2%}",
            help=f"Weighted combination: ML (60%) + Local Fraud Rate (40%)"
        )
    
    with col4:
        # Recommendation with color coding
        recommendation = risk_assessment['recommendation']
        if recommendation == "APPROVE":
            st.success(f"**Recommendation:** {recommendation}")
        elif recommendation == "ESCALATE":
            st.warning(f"**Recommendation:** {recommendation}")
        else:
            st.error(f"**Recommendation:** {recommendation}")
    
    # ------------------------------------------------------------------
    # Risk Level Indicator
    # ------------------------------------------------------------------
    final_score = risk_assessment['final_score']
    if final_score < 0.30:
        risk_level = "🟢 Low Risk"
        risk_color = "green"
    elif final_score < 0.65:
        risk_level = "🟡 Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "🔴 High Risk"
        risk_color = "red"
    
    st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    # ------------------------------------------------------------------
    # SHAP Feature Importance
    # ------------------------------------------------------------------
    with st.expander("🔍 View SHAP Feature Importance (Why this score?)"):
        shap_df = pd.DataFrame(risk_assessment['top_shap_features'])
        
        # Add direction column
        shap_df['direction'] = shap_df['shap_value'].apply(
            lambda x: '↑ Increases Risk' if x > 0 else '↓ Decreases Risk'
        )
        shap_df['abs_impact'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_impact', ascending=False)
        
        # Display formatted table
        st.dataframe(
            shap_df[['feature', 'shap_value', 'direction']].head(10),
            use_container_width=True,
            column_config={
                'feature': 'Feature',
                'shap_value': st.column_config.NumberColumn('Impact', format='%.4f'),
                'direction': 'Effect'
            }
        )
        
        st.caption("Positive SHAP values = pushes prediction toward fraud | Negative = pushes away from fraud")
    
    # ------------------------------------------------------------------
    # Similar Cases Table
    # ------------------------------------------------------------------
    display_similar_cases_table(ret_data["similar_cases"])
    
    # ------------------------------------------------------------------
    # Chat Interface
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("💬 Ask Follow‑up Questions")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input for follow‑up questions
    if prompt := st.chat_input("Ask a follow‑up question (e.g., 'Why was this flagged?', 'What patterns do you see?')"):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer using cached data
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                ret_data = st.session_state.retriever_data
                risk_assessment = st.session_state.risk_assessment
                
                answer = call_writer_agent(
                    ret_data["query_metadata"],
                    ret_data["similar_cases"],
                    ret_data["local_fraud_rate"],
                    risk_assessment,
                    prompt
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_narrative = answer

else:
    st.info("👈 Enter an Application ID in the sidebar and click 'New Analysis' to start.")
    
    # Show example
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Enter an Application ID** in the sidebar (e.g., 856982)
        2. **Click 'New Analysis'** to fetch similar cases and calculate risk score
        3. **Review the risk assessment** including:
           - ML Model Score (60% weight)
           - Local Fraud Rate from similar past cases (40% weight)
           - Final combined risk score
           - SHAP feature importance explaining why
        4. **Ask follow-up questions** in the chat to dive deeper
        5. **Try different Application IDs** to compare risk profiles
        """)