# Final Fraud Detection Test Script
# Uses hybrid rule-based + ML approach
# ----------------------
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List # Added for type hinting

# Load saved artifacts
model = joblib.load("fraud_model_final.pkl")
scaler = joblib.load("scaler_final.pkl")
label_encoders = joblib.load("label_encoders_final.pkl")
feature_cols = joblib.load("feature_cols_final.pkl")

# --- NEUTRAL DEFAULT FALLBACKS (Ensure all features are present) ---
NEUTRAL_NUMERICAL_FALLBACKS: Dict[str, float] = {
    'income': 0.0, 'customer_age': 0.0, 'intended_balcon_amount': 0.0, 'session_length_in_minutes': 0.0,
    'credit_risk_score': 0.0, 'name_email_similarity': 0.0,
    'velocity_6h': 1000.0, 'velocity_24h': 5000.0, 'velocity_4w': 25000.0,
    'prev_address_months_count': 36.0, 'current_address_months_count': 60.0,
    'days_since_request': 10.0, 'zip_count_4w': 5.0, 'bank_branch_count_8w': 3.0,
    'date_of_birth_distinct_emails_4w': 1.0, 'bank_months_count': 36.0,
    'has_other_cards': 0.0, 'proposed_credit_limit': 10000.0, 'foreign_request': 0.0,
    'keep_alive_session': 0.0, 'device_distinct_emails_8w': 1.0, 'device_fraud_count': 0.0,
    'month': float(pd.Timestamp.now().month), 'email_is_free': 0.0, 'phone_home_valid': 1.0, 'phone_mobile_valid': 1.0
}
CATEGORICAL_DEFAULTS: Dict[str, str] = {
    'employment_status': 'employed', 'housing_status': 'owned',
    'payment_type': 'credit_card', 'device_os': 'android', 'source': 'web'
}
# ---------------------------------

# NOTE: The calculate_rule_based_score function is kept from the original for structural alignment
def calculate_rule_based_score(user_data):
    """Calculate rule-based fraud score"""
    # Use .get() with a default for safety, although pre-processing should handle this
    income = user_data.get('income', 0)
    intended_balcon_amount = user_data.get('intended_balcon_amount', 0)
    customer_age = user_data.get('customer_age', 30)
    session_length_in_minutes = user_data.get('session_length_in_minutes', 10)
    credit_risk_score = user_data.get('credit_risk_score', 0)
    name_email_similarity = user_data.get('name_email_similarity', 0.5)
    foreign_request = user_data.get('foreign_request', 0)
    
    # Use 1e-6 to prevent division by zero
    withdrawal_ratio = intended_balcon_amount / (income + 1e-6)
    
    underage_risk = 1 if customer_age < 18 else 0
    extreme_withdrawal = 1 if withdrawal_ratio > 1.0 else 0
    high_withdrawal = 1 if withdrawal_ratio > 0.5 else 0
    very_short_session = 1 if session_length_in_minutes < 1 else 0
    very_long_session = 1 if session_length_in_minutes > 180 else 0
    credit_mismatch = 1 if intended_balcon_amount > credit_risk_score else 0
    low_email_similarity = 1 if name_email_similarity < 0.05 else 0
    foreign_transaction = foreign_request
    
    rule_score = (
        underage_risk * 10 +
        extreme_withdrawal * 8 +
        high_withdrawal * 5 +
        very_short_session * 3 +
        very_long_session * 2 +
        credit_mismatch * 6 +
        low_email_similarity * 3 +
        foreign_transaction * 2
    )
    
    return rule_score, {
        'withdrawal_ratio': withdrawal_ratio,
        'underage_risk': underage_risk,
        'extreme_withdrawal': extreme_withdrawal,
        'high_withdrawal': high_withdrawal,
        'very_short_session': very_short_session,
        'very_long_session': very_long_session,
        'credit_mismatch': credit_mismatch,
        'low_email_similarity': low_email_similarity,
        'foreign_transaction': foreign_transaction
    }

def preprocess_user_input(user_input: Dict[str, Any]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Robust preprocessing: fill defaults, engineer rules, encode, align, and scale.
    """
    
    # 1. INITIALIZE & FILL MISSING VALUES
    processed_data = {}
    
    # Fill Numerical/Binary Features with explicit fallbacks
    for field, fallback_value in NEUTRAL_NUMERICAL_FALLBACKS.items():
        if field in user_input:
            try:
                processed_data[field] = float(user_input[field])
            except (ValueError, TypeError):
                processed_data[field] = fallback_value
        else:
            processed_data[field] = fallback_value

    # Fill Categorical Features with explicit defaults
    for field, default_value in CATEGORICAL_DEFAULTS.items():
        processed_data[field] = str(user_input.get(field, default_value))
    
    
    # 2. RULE-BASED FEATURE ENGINEERING (Features for ML vector)
    rule_score, rule_flags = calculate_rule_based_score(processed_data)
    processed_data.update(rule_flags)
    
    # Add other required engineered binary features for the ML vector
    processed_data['high_velocity_6h'] = 1.0 if processed_data.get('velocity_6h', 0) > 10000 else 0.0 
    processed_data['high_velocity_24h'] = 1.0 if processed_data.get('velocity_24h', 0) > 50000 else 0.0
    processed_data['device_fraud_history'] = 1.0 if processed_data.get('device_fraud_count', 0) > 0 else 0.0
    processed_data['rule_based_fraud_score'] = float(rule_score)
    
    
    # 3. CONVERT TO DATAFRAME FOR ENCODING/ALIGNMENT
    df = pd.DataFrame([processed_data])

    # 4. LABEL ENCODE CATEGORICALS (using saved encoders)
    for col, encoder in label_encoders.items():
        if col in df.columns:
            # Safely encode, setting unknown categories to 0.0
            df[col] = df[col].apply(
                lambda x: float(encoder.transform([str(x)])[0]) if str(x) in encoder.classes_ else 0.0
            )
        elif col not in df.columns:
             df[col] = 0.0


    # 5. ALIGN FINAL FEATURE VECTOR AND SCALE (CRITICAL STEP)
    for col in feature_cols: 
        if col not in df.columns:
            df[col] = 0.0
            
    # Extract the vector in the PRECISE ORDER expected by the model
    feature_vector_aligned = df[feature_cols].values.astype(float) 

    # Scale the aligned feature vector
    feature_scaled = scaler.transform(feature_vector_aligned)
    
    # Return scaled vector, final score, and flags for API response
    return feature_scaled, rule_score, rule_flags


def hybrid_predict(user_input):
    """Hybrid prediction: Rules first, then ML for edge cases, returns safety confidence"""
    
    df_scaled, rule_score, risk_factors = preprocess_user_input(user_input)
    
    ml_pred = model.predict(df_scaled)[0]
    ml_prob = model.predict_proba(df_scaled)[0][1] # Probability of Fraud

    # Calculate Safety Probability (requested by user)
    safety_prob = 1.0 - ml_prob
    
    # Hybrid Decision Logic
    if rule_score >= 5:
        hybrid_pred = 1
        prediction_method = "Rule-based (High Risk Score)"
    elif ml_prob > 0.7:
        hybrid_pred = 1
        prediction_method = "ML High Confidence"
    else:
        # FIX: Implement a custom threshold (e.g., 0.65) higher than the default 0.5
        # This prevents safe transactions with ML Prob > 0.5 but < 0.7 from being flagged as Fraud.
        if ml_prob > 0.65:
            hybrid_pred = 1
            prediction_method = "ML Standard (Custom Fraud Threshold)"
        else:
            hybrid_pred = 0
            prediction_method = "ML Standard"
    
    return hybrid_pred, ml_prob, rule_score, prediction_method, risk_factors, safety_prob


def analyze_risk_factors(risk_factors, rule_score):
    """Analyze which factors contribute to fraud risk"""
    risk_analysis = []
    
    if risk_factors.get('underage_risk'):
        risk_analysis.append("🚨 CRITICAL: Underage customer (age < 18)")
    
    if risk_factors.get('extreme_withdrawal'):
        risk_analysis.append("🚨 CRITICAL: Withdrawal exceeds income")
    
    if risk_factors.get('high_withdrawal'):
        risk_analysis.append("🚨 HIGH: Withdrawal > 50% of income")
    
    if risk_factors.get('very_short_session'):
        risk_analysis.append("⚠️ SUSPICIOUS: Very short session (< 1 minute)")
    
    if risk_factors.get('very_long_session'):
        risk_analysis.append("⚠️ SUSPICIOUS: Very long session (> 3 hours)")
    
    if risk_factors.get('credit_mismatch'):
        risk_analysis.append("⚠️ HIGH: Withdrawal exceeds credit risk score")
    
    if risk_factors.get('low_email_similarity'):
        risk_analysis.append("⚠️ SUSPICIOUS: Low email similarity")
    
    if risk_factors.get('foreign_transaction'):
        risk_analysis.append("⚠️ MODERATE: Foreign transaction")
    
    ratio = risk_factors.get('withdrawal_ratio', 0)
    
    # Update analysis to emphasize safety for low ratios
    if ratio < 0.01:
         risk_analysis.append(f"✅ VERY LOW RISK: Withdrawal is only {ratio:.2%} of income.")
    elif ratio > 1.0:
        risk_analysis.append(f"🚨 EXTREME: Withdrawal is {ratio:.1f}x income")
    elif ratio > 0.5:
        risk_analysis.append(f"⚠️ HIGH: Withdrawal is {ratio:.1%} of income")
    elif ratio > 0.1:
        risk_analysis.append(f"ℹ️ MODERATE: Withdrawal is {ratio:.1%} of income")
    
    return risk_analysis

# Main interactive section
if __name__ == "__main__":
    # Injected neutral values for local testing completeness
    NEUTRAL_TEST_INPUTS = {
        'credit_risk_score': 5000,
        'name_email_similarity': 0.9,
        **{k: v for k, v in NEUTRAL_NUMERICAL_FALLBACKS.items() if k not in ['income', 'customer_age', 'intended_balcon_amount', 'session_length_in_minutes', 'credit_risk_score', 'name_email_similarity']},
        **CATEGORICAL_DEFAULTS
    }

    print("=" * 70)
    print("🔍 FINAL FRAUD DETECTION SYSTEM")
    print("   Hybrid Rule-Based + Machine Learning")
    print("=" * 70)
    
    user_data = {}
    
    print("\n📝 Please provide transaction details (enter default if unsure):")
    # Collect core user inputs
    user_data['income'] = input("Enter income (e.g., 50000): ")
    user_data['customer_age'] = input("Enter customer age: ")
    user_data['intended_balcon_amount'] = input("Enter intended withdrawal amount (e.g., 500): ")
    user_data['session_length_in_minutes'] = input("Enter session length in minutes: ")
    user_data['credit_risk_score'] = input("Enter credit risk score (e.g., 5000): ")
    user_data['name_email_similarity'] = input("Enter email similarity (e.g., 0.9): ")
    user_data['payment_type'] = input("Enter payment type (credit_card, debit_card, paypal): ")
    user_data['device_os'] = input("Enter device OS (ios, android, windows): ")
    
    # Merge with neutral defaults for features not requested from user
    final_user_data = {**NEUTRAL_TEST_INPUTS, **user_data}
    
    # Updated call to hybrid_predict
    prediction, ml_prob, rule_score, method, risk_factors, safety_prob = hybrid_predict(final_user_data)
    risk_analysis = analyze_risk_factors(risk_factors, rule_score)
    
    print(f"\n" + "=" * 70)
    print("🎯 FRAUD ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Final Prediction: {'🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE'}")
    print(f"Prediction Method: {method}")
    print(f"Rule-based Score: {rule_score}/10")
    print(f"ML Probability of Fraud: {ml_prob:.4f}")
    
    # EMPHASIZE SAFETY CONFIDENCE
    print(f"Confidence of **Safety**: **{safety_prob:.2%}** ({safety_prob:.4f})") 
    
    if risk_analysis:
        print(f"\n🚨 RISK FACTORS DETECTED:")
        for factor in risk_analysis:
            print(f"   {factor}")
    else:
        print(f"\n✅ No significant risk factors detected")
    
    print(f"\n" + "=" * 70)
    if prediction == 1:
        print("🚨 FRAUD DETECTED! Transaction blocked.")
        if rule_score >= 5:
            print("   Reason: Rule-based high-risk indicators detected")
        else:
            print("   Reason: Machine learning high-confidence fraud detection")
    else:
        print("✅ Legitimate transaction. Proceeding...")
        print("   Reason: High confidence in transaction safety.")
    print("=" * 70)