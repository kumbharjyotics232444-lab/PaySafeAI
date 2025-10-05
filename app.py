from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import pandas as pd
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import certifi
import numpy as np
from bson import ObjectId

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "yoursecretkey"

# -----------------------------
# MongoDB Atlas connection
# -----------------------------
MONGO_URI = "mongodb+srv://kumbharjyotics232444:2rJuzrAMbUS9ZtQB@cluster0.rlkth.mongodb.net/PaySafeAI?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["PaySafeAI"]
users_col = db["users"]
history_col = db["history"]

# -----------------------------
# Page Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login.html")
def login_page():
    return render_template("login.html")

@app.route("/signup.html")
def signup_page():
    return render_template("signup.html")

@app.route("/index_user.html")
def index_user():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("index_user.html")

@app.route("/index_admin.html")
def index_admin():
    if "user" not in session:
        return redirect(url_for("login_page"))
    if session.get("role") != "admin":
        return redirect(url_for("index_user"))
    return render_template("index_admin.html")

@app.route("/user_profile.html")
def user_profile():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("user_profile.html")

@app.route("/user_history.html")
def user_history():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("user_history.html", logged_in_user=session["user"])

@app.route("/user_transaction_form.html")
def user_transaction_form():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("user_transaction_form.html")

@app.route("/admin_user.html")
def admin_user():
    if "user" not in session or session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("admin_user.html")

@app.route("/admin_dashboard.html")
def admin_dashboard():
    if "user" not in session or session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("admin_dashboard.html")

# -----------------------------
# API Routes
# -----------------------------
from test_final import hybrid_predict

@app.route("/api/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json or {}
        prediction, ml_prob, rule_score, method, risk_factors, safety_prob = hybrid_predict(data)
        result = "Fraud" if prediction == 1 else "Safe"

        history_col.insert_one({
            "username": session["user"],
            **data,
            "prediction": result,
            "ml_probability": float(ml_prob),
            "rule_score": int(rule_score),
            "method": method,
            "safety_probability": float(safety_prob)
        })

        return jsonify({
            "prediction": result,
            "probability": float(ml_prob),
            "safety_probability": float(safety_prob),
            "rule_score": int(rule_score),
            "method": method,
            "risk_factors": risk_factors
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if users_col.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(password)
    users_col.insert_one({
        "name": name,
        "email": email,
        "password": hashed_pw,
        "role": "user"
    })
    return jsonify({"message": "Signup successful"}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users_col.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        session["user"] = email
        session["role"] = user.get("role", "user")
        return jsonify({
            "message": "Login successful",
            "user": {"email": user["email"], "role": session["role"]}
        })
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/user/<email>", methods=["GET", "PUT"])
def user_profile_api(email):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if request.method == "GET":
        return jsonify({"name": user.get("name"), "email": user.get("email"), "role": user.get("role", "user")})

    if request.method == "PUT":
        data = request.json
        update_data = {
            "name": data.get("name", user.get("name")),
            "email": data.get("email", user.get("email"))
        }
        if data.get("password"):
            update_data["password"] = generate_password_hash(data.get("password"))

        users_col.update_one({"email": email}, {"$set": update_data})
        session["user"] = update_data["email"]
        return jsonify({"user": update_data})

# -----------------------------
# Fixed: Include _id as string
# -----------------------------
@app.route("/api/history", methods=["GET"])
def history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    records = list(history_col.find({"username": session["user"]}))
    for rec in records:
        rec["_id"] = str(rec["_id"])
    return jsonify(records)

@app.route("/api/history/<id>", methods=["DELETE"])
def delete_transaction(id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        result = history_col.delete_one({"_id": ObjectId(id), "username": session["user"]})
        if result.deleted_count == 0:
            return jsonify({"error": "Transaction not found or not allowed"}), 404
        return jsonify({"message": "Transaction deleted"})
    except:
        return jsonify({"error": "Invalid transaction ID"}), 400

@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    session.pop("role", None)
    return jsonify({"message": "Logged out successfully"})

# -----------------------------
# Admin APIs
# -----------------------------
@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    if "user" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    users = list(users_col.find({}, {"name": 1, "email": 1, "role": 1}))
    for u in users:
        u["_id"] = str(u["_id"])
    return jsonify(users)

@app.route("/api/admin/users/<id>", methods=["DELETE"])
def delete_user(id):
    if "user" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    users_col.delete_one({"_id": ObjectId(id)})
    return jsonify({"message": "User deleted"})

@app.route("/api/admin/dashboard", methods=["GET"])
def admin_dashboard_api():
    if "user" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    total_tx = history_col.count_documents({})
    fraud_tx = history_col.count_documents({"prediction": "Fraud"})
    safe_tx = history_col.count_documents({"prediction": "Safe"})
    total_users = users_col.count_documents({})

    return jsonify({
        "total": total_tx,
        "frauds": fraud_tx,
        "safe": safe_tx,
        "users": total_users,
        "model_version": "v1.0"
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
