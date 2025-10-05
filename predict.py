import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

try:
    model = joblib.load(os.path.join(BASE_DIR, "fraud_model_final.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_final.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders_final.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols_final.pkl"))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Failed to load model artifact: {e}")


# HELPER FUNCTIONS 

def calculate_rule_based_score(user_data):
    """Calculate rule-based fraud score and derived features/flags"""
    
    # Calculate Ratios
    income = user_data.get('income', 0.0)
    withdrawal = user_data.get('intended_balcon_amount', 0.0)
    credit_risk_score = user_data.get('credit_risk_score', 0.0)
    
    withdrawal_ratio = withdrawal / (income + 1)
    
    # Rule Flags 
    underage_risk = 1 if user_data.get('customer_age', 30) < 18 else 0
    extreme_withdrawal = 1 if withdrawal_ratio > 1.0 else 0
    high_withdrawal = 1 if withdrawal_ratio > 0.5 else 0
    very_short_session = 1 if user_data.get('session_length_in_minutes', 10) < 1 else 0
    very_long_session = 1 if user_data.get('session_length_in_minutes', 10) > 180 else 0
    credit_mismatch = 1 if withdrawal > credit_risk_score else 0
    low_email_similarity = 1 if user_data.get('name_email_similarity', 0.5) < 0.05 else 0
    
    foreign_transaction = user_data.get('foreign_request', 0)
    
    # Calculate Score
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
        'withdrawal_income_ratio': withdrawal_ratio,
        'underage_risk': underage_risk,
        'extreme_withdrawal': extreme_withdrawal,
        'high_withdrawal': high_withdrawal,
        'very_short_session': very_short_session,
        'very_long_session': very_long_session,
        'credit_mismatch': credit_mismatch,
        'low_email_similarity': low_email_similarity,
        'foreign_transaction': foreign_transaction
    }


def analyze_risk_factors(rule_based_flags, rule_score):
    risk_analysis = []
    #rule based flags
    if rule_based_flags['underage_risk']:
        risk_analysis.append("CRITICAL: Underage customer (age < 18)")
    
    if rule_based_flags['extreme_withdrawal']:
        risk_analysis.append("CRITICAL: Withdrawal exceeds income")
    
    if rule_based_flags['high_withdrawal']:
        risk_analysis.append("HIGH: Withdrawal > 50% of income")
    
    if rule_based_flags['very_short_session']:
        risk_analysis.append("SUSPICIOUS: Very short session (< 1 minute)")
    
    if rule_based_flags['very_long_session']:
        risk_analysis.append("SUSPICIOUS: Very long session (> 3 hours)")
    
    if rule_based_flags['credit_mismatch']:
        risk_analysis.append("HIGH: Withdrawal exceeds credit risk score")
    
    if rule_based_flags['low_email_similarity']:
        risk_analysis.append("SUSPICIOUS: Low email similarity")
    
    if rule_based_flags['foreign_transaction']:
        risk_analysis.append("MODERATE: Foreign transaction")
    
    ratio = rule_based_flags['withdrawal_income_ratio']
    if ratio > 1.0:
        risk_analysis.append(f"EXTREME: Withdrawal is {ratio:.1f}x income")
    elif ratio > 0.5:
        risk_analysis.append(f"HIGH: Withdrawal is {ratio:.1%} of income")
    
    return risk_analysis


def preprocess_user_input(user_input):
    
    # Converts numeric fields s
    numeric_fields = [
        'income', 'intended_balcon_amount', 'customer_age', 'session_length_in_minutes',
        'credit_risk_score', 'name_email_similarity', 'velocity_6h', 'velocity_24h', 
        'velocity_4w', 'prev_address_months_count', 'current_address_months_count',
        'days_since_request', 'zip_count_4w', 'bank_branch_count_8w',
        'date_of_birth_distinct_emails_4w', 'bank_months_count',
        'has_other_cards', 'proposed_credit_limit', 'foreign_request',
        'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count',
        'month', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid' # Binary flags
    ]
    
    for field in numeric_fields:
        if field in user_input:
            try:
                user_input[field] = float(user_input[field])
            except (ValueError, TypeError):
                user_input[field] = 0.0
        else:
            user_input[field] = 0.0
    
    categorical_defaults = {
        'employment_status': 'employed', 'housing_status': 'owned', 
        'payment_type': 'credit_card', 'device_os': 'android', 'source': 'web'
    }
    for key, val in categorical_defaults.items():
        if key not in user_input:
            user_input[key] = val
        user_input[key] = str(user_input[key])

    #  Calculate rule-based indicators and the score
    rule_score, rule_based_flags = calculate_rule_based_score(user_input)
    
    #  Update user_input with all derived features
    user_input.update(rule_based_flags)
    user_input['rule_based_fraud_score'] = rule_score
    
    #  Final Feature Vector Alignment
    feature_vector = [user_input.get(col, 0.0) for col in feature_cols]

    # Convert to DataFrame for final encoding/scaling step
    df = pd.DataFrame([feature_vector], columns=feature_cols)

    #  Label Encode Categorical Variables
    categorical_cols_to_encode = list(label_encoders.keys()) # The keys are the columns to encode
    for col in categorical_cols_to_encode:
        if col in df.columns and col in user_input:
            try:
                df[col] = label_encoders[col].transform([user_input[col]])
            except ValueError:
                df[col] = 0
        elif col in df.columns:
             df[col] = 0 
            
    #  Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled, rule_score, rule_based_flags


def hybrid_predict(user_input):
    
    # 1. Preprocess and get feature vector
    df_scaled, rule_score, rule_based_flags = preprocess_user_input(user_input)
    
    # 2. ML Prediction
    ml_prob = model.predict_proba(df_scaled)[0][1]
    ml_pred = model.predict(df_scaled)[0]
    
    # 3. Hybrid Decision Logic (Rule-based score overrides ML)
    if rule_score >= 5:
        hybrid_pred = 1
        prediction_method = "Rule-based (High Risk Score)"
    elif ml_prob > 0.7: # High confidence ML overrides standard ML threshold
        hybrid_pred = 1
        prediction_method = "ML High Confidence"
    else:
        # Use standard ML prediction if no rule violation
        hybrid_pred = ml_pred
        prediction_method = "ML Standard"
    
    # 4. Risk Analysis
    risk_analysis = analyze_risk_factors(rule_based_flags, rule_score)

    return hybrid_pred, ml_prob, rule_score, prediction_method, risk_analysis



def predict_transaction(data):
    
    try:
        hybrid_pred, ml_prob, rule_score, method, risk_factors = hybrid_predict(data)
        
        return {
            "prediction": "Fraud" if hybrid_pred == 1 else "Safe",
            "probability": float(ml_prob),
            "rule_score": int(rule_score),
            "method": method,
            "risk_factors": risk_factors
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {
            "error": "Prediction failed due to internal server error.",
            "details": str(e)
        }
