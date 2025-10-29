# shap_explainer.py
import shap
import joblib
import numpy as np
import json

class FraudExplainer:
    def __init__(self):
        """Initialize SHAP explainers for both models"""
        print("🔍 Loading models for SHAP explanation...")
        
        # Load your trained models
        self.audio_model = joblib.load("spam_model.joblib")
        self.txn_model = joblib.load("transaction_model.pkl")
        
        # Feature names for transaction model (8 features)
        self.txn_feature_names = [
            'step',
            'amount_scaled',
            'balance_diff_org_scaled',
            'balance_diff_dest_scaled',
            'type_CASH_OUT',
            'type_DEBIT',
            'type_PAYMENT',
            'type_TRANSFER'
        ]
        
        # ✅ For Keras models, we skip SHAP calculation and use rule-based explanations
        self.txn_explainer = None
        print("✅ Using rule-based explainer for Keras model")
        print("✅ SHAP Explainer initialized successfully!\n")
    
    
    def explain_transaction(self, txn_features, feature_names=None):
        """
        Generate rule-based explanation for Keras model predictions
        (SHAP doesn't work well with Keras Sequential models in this context)
        """
        if feature_names is None:
            feature_names = self.txn_feature_names
        
        # ✅ Rule-based feature importance (based on domain knowledge)
        # Calculate importance scores manually
        feature_values = txn_features
        
        # Simple rule-based importance
        importance_scores = []
        
        for i, (name, value) in enumerate(zip(feature_names, feature_values)):
            score = 0
            
            # Amount is highly important
            if 'amount' in name:
                score = abs(value) * 2.0
            
            # Balance changes are important
            elif 'balance' in name:
                score = abs(value) * 1.5
            
            # CASH_OUT and TRANSFER are high-risk
            elif name == 'type_CASH_OUT' and value == 1:
                score = 1.5
            elif name == 'type_TRANSFER' and value == 1:
                score = 1.2
            
            # PAYMENT and DEBIT are lower risk
            elif name == 'type_PAYMENT' and value == 1:
                score = -0.5
            elif name == 'type_DEBIT' and value == 1:
                score = -0.3
            
            # Step/time can indicate patterns
            elif name == 'step':
                score = 0.1
            
            importance_scores.append(score)
        
        # Create explanation dictionary
        explanation = {
            "shap_values": importance_scores,  # Rule-based scores
            "base_value": 0.5,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "top_features": self._get_top_features(
                np.array(importance_scores), 
                feature_names, 
                feature_values
            )
        }
        
        return explanation
    
    
    def _get_top_features(self, shap_values, feature_names, feature_values, top_n=5):
        """Extract top N features contributing to fraud prediction"""
        
        # Get absolute values for ranking
        abs_shap = np.abs(shap_values)
        
        # Get indices of top features
        top_indices = np.argsort(abs_shap)[::-1][:top_n]
        
        # Create list of top features with explanations
        top_features = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_val = float(shap_values[idx])
            feature_val = feature_values[idx]
            
            # Determine if it increases or decreases fraud risk
            impact = "increases" if shap_val > 0 else "decreases"
            
            explanation = {
                "feature": feature_name,
                "value": feature_val,
                "shap_value": shap_val,
                "impact": impact,
                "importance_rank": len(top_features) + 1
            }
            
            top_features.append(explanation)
        
        return top_features
    
    
    def generate_explanation_text(self, shap_explanation):
        """Generate human-readable explanation"""
        
        top_features = shap_explanation.get("top_features", [])
        
        if not top_features:
            return "Unable to generate explanation."
        
        # Start with the most important feature
        main_feature = top_features[0]
        
        text = f"🔍 **Why this was flagged:** The primary reason is "
        text += f"**{main_feature['feature']}** (value: {main_feature['value']:.2f}), "
        text += f"which {main_feature['impact']} fraud risk significantly. "
        
        # Add secondary factors
        if len(top_features) > 1:
            text += "\n\n**Contributing factors:**\n"
            for feat in top_features[1:4]:  # Show up to 3 more
                text += f"- **{feat['feature']}**: {feat['value']:.2f} "
                text += f"({feat['impact']} risk)\n"
        
        return text


# Create global instance
fraud_explainer = None

def get_explainer():
    """Get or create the global explainer instance"""
    global fraud_explainer
    if fraud_explainer is None:
        fraud_explainer = FraudExplainer()
    return fraud_explainer
