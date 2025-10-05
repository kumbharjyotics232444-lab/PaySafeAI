import joblib
import pandas as pd
import numpy as np
import os

# --- 1. SETUP & ARTIFACT LOADING ---

# Use the directory where this script is located for relative pathing (Recommended over hardcoded paths)
# If your PKL files are in a different, fixed location, adjust this path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Load FINAL model and artifacts. NOTE THE '_final' SUFFIX.
try:
    model = joblib.load(os.path.join(BASE_DIR, "fraud_model_final.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_final.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders_final.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols_final.pkl"))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Failed to load model artifact: {e}")
    # In a real app, you would log this and return a 500 error, but for development, we print.


# --- 2. HELPER FUNCTIONS (COPIED FROM test_final.py) ---

def calculate_rule_based_score(user_data):
    """Calculate rule-based fraud score and derived features/flags"""
    
    # Calculate Ratios
    # Use .get() and safe denominator for robustness, assuming 0.0 is the default safe value
    income = user_data.get('income', 0.0)
    withdrawal = user_data.get('intended_balcon_amount', 0.0)
    credit_risk_score = user_data.get('credit_risk_score', 0.0)
    
    withdrawal_ratio = withdrawal / (income + 1)
    
    # Rule Flags (1 = Rule Violated/High Risk)
    underage_risk = 1 if user_data.get('customer_age', 30) < 18 else 0
    extreme_withdrawal = 1 if withdrawal_ratio > 1.0 else 0
    high_withdrawal = 1 if withdrawal_ratio > 0.5 else 0
    very_short_session = 1 if user_data.get('session_length_in_minutes', 10) < 1 else 0
    very_long_session = 1 if user_data.get('session_length_in_minutes', 10) > 180 else 0
    credit_mismatch = 1 if withdrawal > credit_risk_score else 0
    low_email_similarity = 1 if user_data.get('name_email_similarity', 0.5) < 0.05 else 0
    
    # Use provided values from the full payload (default 0 if system defaults fail)
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
    
    # Return features for further preprocessing/logging
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
    """Analyze which factors contribute to fraud risk"""
    risk_analysis = []
    
    # Use rule_based_flags dictionary for analysis
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
    
    # Add ratio detail for better context
    ratio = rule_based_flags['withdrawal_income_ratio']
    if ratio > 1.0:
        risk_analysis.append(f"EXTREME: Withdrawal is {ratio:.1f}x income")
    elif ratio > 0.5:
        risk_analysis.append(f"HIGH: Withdrawal is {ratio:.1%} of income")
    
    return risk_analysis


def preprocess_user_input(user_input):
    """Preprocess user input safely for hybrid fraud detection"""
    
    # 1️⃣ Convert numeric fields safely (Including all 23 features + user inputs)
    # This list is comprehensive to handle all fields in the 'feature_cols' up to 'month'
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
                # Use float for all numeric/binary inputs
                user_input[field] = float(user_input[field])
            except (ValueError, TypeError):
                user_input[field] = 0.0
        else:
            # Ensure missing system defaults are added as 0.0
            user_input[field] = 0.0
    
    # 2️⃣ Set default values for missing categorical features
    # These are used for encoding
    categorical_defaults = {
        'employment_status': 'employed', 'housing_status': 'owned', 
        'payment_type': 'credit_card', 'device_os': 'android', 'source': 'web'
    }
    for key, val in categorical_defaults.items():
        if key not in user_input:
            user_input[key] = val
        # Ensure they are strings for the encoder
        user_input[key] = str(user_input[key])

    # 3️⃣ Calculate rule-based indicators and the score
    rule_score, rule_based_flags = calculate_rule_based_score(user_input)
    
    # 4️⃣ Update user_input with all derived features
    user_input.update(rule_based_flags)
    user_input['rule_based_fraud_score'] = rule_score
    
    # 5️⃣ Final Feature Vector Alignment
    feature_vector = [user_input.get(col, 0.0) for col in feature_cols]

    # Convert to DataFrame for final encoding/scaling step
    df = pd.DataFrame([feature_vector], columns=feature_cols)

    # 6️⃣ Label Encode Categorical Variables
    # Note: We must encode the dataframe, not the dictionary, for correct column alignment
    categorical_cols_to_encode = list(label_encoders.keys()) # The keys are the columns to encode
    for col in categorical_cols_to_encode:
        if col in df.columns and col in user_input:
            # Use a helper function for safe transform (handling unknown categories)
            try:
                df[col] = label_encoders[col].transform([user_input[col]])
            except ValueError:
                # Assign to 0 (or a known 'Other' category if your encoder was fit that way)
                df[col] = 0
        elif col in df.columns:
             # Should not happen if step 2/3 are correct, but handles unexpected missing categorical
             df[col] = 0 
            
    # 7️⃣ Scale features
    df_scaled = scaler.transform(df)
    
    # Return all necessary components for prediction and analysis
    return df_scaled, rule_score, rule_based_flags


def hybrid_predict(user_input):
    """Hybrid prediction: Rules first, then ML for edge cases"""
    
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


# --- 3. THE NEW API ENDPOINT FUNCTION ---

def predict_transaction(data):
    """
    Predict if a transaction is 'Fraud' or 'Safe' using the Hybrid Model.
    """
    # NOTE: The data.pop() block has been permanently REMOVED to allow all data through.
    
    try:
        # Run the full hybrid prediction logic
        hybrid_pred, ml_prob, rule_score, method, risk_factors = hybrid_predict(data)
        
        # Format the result as expected by the frontend JavaScript
        return {
            "prediction": "Fraud" if hybrid_pred == 1 else "Safe",
            "probability": float(ml_prob),
            "rule_score": int(rule_score),
            "method": method,
            "risk_factors": risk_factors
        }

    except Exception as e:
        # Catch any errors during prediction or feature engineering
        print(f"Prediction Error: {e}")
        return {
            "error": "Prediction failed due to internal server error.",
            "details": str(e) # Useful for debugging, remove for production
        }

# If your Flask app is in a separate file, you just need to ensure
# it imports and calls this 'predict_transaction' function in its API route.
# For example, in app.py:
"""
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    result = predict_transaction(data)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)
"""