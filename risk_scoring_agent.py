"""
risk_scoring_agent.py
Combines ML model fraud score with retriever local fraud rate and SHAP values
"""

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from src.preprocessing.baf_preprocessor import BAFPreprocessor, TimeSplit
from src.retriever.enrichment import build_retriever_features_for_records


class RiskScoringAgent:
    """
    Combines multiple signals into a final risk score:
    - ML model fraud probability (from XGBoost)
    - Retriever local fraud rate (from similar past cases)
    - SHAP values for explanation
    """
    
    def __init__(self, 
                 model_path="results/enriched/model.pkl",
                 weights=None):
        """
        Initialize the risk scoring agent.
        
        Args:
            model_path: Path to trained XGBoost model
            weights: Dict with weights for 'ml_score' and 'local_fraud_rate'
                    Default: {'ml_score': 0.6, 'local_fraud_rate': 0.4}
        """
        self.weights = weights or {'ml_score': 0.6, 'local_fraud_rate': 0.4}
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Create preprocessor (same logic as training)
        print("Creating preprocessor...")
        self.preprocessor = BAFPreprocessor()
        split = TimeSplit()
        
        # Load data to fit preprocessor
        df = pd.read_csv("data/Base.csv")
        train_df, valid_df, test_df = self.preprocessor.split_by_month(df, split)
        self.preprocessor.fit(train_df)
        
        # Create SHAP explainer
        print("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"✅ Risk Scoring Agent initialized")
        print(f"   Weights: ML={self.weights['ml_score']}, Retriever={self.weights['local_fraud_rate']}")
    
    def get_ml_score(self, application_dict):
        """Get fraud probability from ML model."""
        # Convert to DataFrame
        df = pd.DataFrame([application_dict])
        
        # Preprocess
        X = self.preprocessor.transform_features(df)
        
        # Add retriever features (if available in the dict)
        # For now, use zeros if not provided
        retriever_cols = ['retr_local_fraud_rate', 'retr_total_neighbors', 
                          'retr_fraud_neighbors', 'retr_similarity_mean', 'retr_similarity_max']
        
        for col in retriever_cols:
            if col not in X.columns:
                X[col] = 0.0
        
        # Get prediction
        ml_score = float(self.model.predict_proba(X)[:, 1][0])
        
        # Get SHAP values for explanation
        shap_values = self.explainer.shap_values(X)
        
        # Get top features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return ml_score, feature_importance
    
    def compute_final_score(self, ml_score, local_fraud_rate):
        """
        Combine ML score and local fraud rate with configured weights.
        
        Args:
            ml_score: Fraud probability from ML model (0-1)
            local_fraud_rate: Fraud rate among similar past cases (0-1)
        
        Returns:
            Final risk score (0-1)
        """
        final_score = (
            self.weights['ml_score'] * ml_score +
            self.weights['local_fraud_rate'] * local_fraud_rate
        )
        return min(max(final_score, 0.0), 1.0)
    
    def get_recommendation(self, final_score, thresholds=None):
        """Get recommendation based on final score."""
        thresholds = thresholds or {'approve': 0.30, 'escalate': 0.65}
        
        if final_score < thresholds['approve']:
            return "APPROVE"
        elif final_score < thresholds['escalate']:
            return "ESCALATE"
        else:
            return "REJECT"
    
    def assess_application(self, application_dict, retriever_output):
        """
        Complete assessment of an application.
        
        Args:
            application_dict: Raw application features
            retriever_output: Dict from retriever agent containing:
                - local_fraud_rate
                - similar_cases
                - total_neighbors
        
        Returns:
            Dict with complete assessment
        """
        # Get ML score and SHAP
        ml_score, shap_features = self.get_ml_score(application_dict)
        
        # Get retriever local fraud rate
        local_fraud_rate = retriever_output.get('local_fraud_rate', 0.0)
        
        # Compute final score
        final_score = self.compute_final_score(ml_score, local_fraud_rate)
        
        # Get recommendation
        recommendation = self.get_recommendation(final_score)
        
        # Prepare explanation
        explanation = {
            'ml_score': ml_score,
            'local_fraud_rate': local_fraud_rate,
            'final_score': final_score,
            'weights_used': self.weights,
            'top_shap_features': shap_features.head(10).to_dict('records'),
            'recommendation': recommendation,
            'similar_cases_count': retriever_output.get('total_neighbors', 0)
        }
        
        return explanation


# Standalone function for easy integration with existing code
def calculate_risk_score(application_dict, retriever_output, weights=None):
    """
    Convenience function to calculate risk score.
    
    Args:
        application_dict: Raw application features
        retriever_output: Output from retriever agent (/retrieve endpoint)
        weights: Optional custom weights
    
    Returns:
        Dict with risk assessment
    """
    agent = RiskScoringAgent(weights=weights)
    return agent.assess_application(application_dict, retriever_output)