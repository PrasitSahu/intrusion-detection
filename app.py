import os
import random
import sqlite3
from datetime import datetime
import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, render_template, request, jsonify

from train import engineer_features, cols, categorize, ATTACK_CATEGORIES

app = Flask(__name__)

# Database setup
DB_NAME = 'ids_logs.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            true_label TEXT,
            predicted_label TEXT,
            attack_type TEXT,
            is_correct BOOLEAN,
            features_json TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE request_logs ADD COLUMN attack_type TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

# Load GPU-trained models
print("Loading GPU-trained XGBoost models...")
binary_model = xgb.XGBClassifier()
binary_model.load_model('models/binary_xgb_gpu.json')

artifacts = joblib.load('models/threshold.pkl')
threshold = artifacts['threshold'] if isinstance(artifacts, dict) else artifacts
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

multiclass_model = xgb.XGBClassifier()
multiclass_model.load_model('models/multiclass_xgb_gpu.json')
label_encoder = joblib.load('models/label_encoder.pkl')

print("Models loaded successfully.")
print(f"  Binary threshold: {threshold}")
print(f"  Feature count: {len(model_columns)}")

cols = ["duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty_level"]

print("Loading test data into memory for traffic simulation...")
df_test = pd.read_csv("data/KDDTest+.txt", names=cols)
df_test['category'] = df_test['label'].apply(categorize)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/send_traffic', methods=['POST'])
def send_traffic():
    try:
        req_type = request.json.get('type', 'any')
        category = request.json.get('category', '').lower()
        
        if category and category != 'any':
            matched = df_test['category'].str.lower() == category
            filtered_df = df_test[matched]
        elif req_type == 'normal':
            filtered_df = df_test[df_test['label'] == 'normal']
        elif req_type == 'attack':
            filtered_df = df_test[df_test['label'] != 'normal']
        else:
            filtered_df = df_test
            
        if filtered_df.empty:
            return jsonify({"status": "error", "message": f"No samples for category '{category}'"}), 400
        row = filtered_df.sample(1).iloc[0]
        true_label = 'normal' if row['label'] == 'normal' else 'attack'
        
        packet_data = {}
        for col in cols:
            if col not in ['label', 'difficulty_level']:
                packet_data[col] = str(row[col])
        
        df_input = pd.DataFrame([packet_data])
        numeric_cols = [c for c in df_input.columns if c not in ['protocol_type', 'service', 'flag']]
        for col in numeric_cols:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)
        df_input = engineer_features(df_input)
        
        categorical_cols = ['protocol_type', 'service', 'flag', 'protocol_service', 'protocol_flag']
        df_input = pd.get_dummies(df_input, columns=categorical_cols)
        bool_cols = df_input.select_dtypes('bool').columns.tolist()
        if bool_cols:
            df_input[bool_cols] = df_input[bool_cols].astype(int)
        df_input = df_input.reindex(columns=model_columns, fill_value=0)
        
        X_scaled = scaler.transform(df_input.values)
        df_input = pd.DataFrame(X_scaled, columns=df_input.columns)
        
        p_attack = binary_model.predict_proba(df_input)[0, 1]
        is_attack = p_attack >= threshold
        prediction = 'attack' if is_attack else 'normal'
        
        attack_type = ''
        if is_attack and multiclass_model is not None and label_encoder is not None:
            attack_probs = multiclass_model.predict_proba(df_input)[0]
            attack_type_idx = np.argmax(attack_probs)
            attack_type = label_encoder.classes_[attack_type_idx]
            if attack_type == 'normal':
                prediction = 'normal'
                attack_type = ''
        
        is_correct = bool(prediction == true_label)
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO request_logs (timestamp, true_label, predicted_label, attack_type, is_correct, features_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), true_label, prediction, attack_type, is_correct, json.dumps(packet_data)))
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Traffic sent to network.",
            "type_sent": true_label,
            "predicted_label": prediction,
            "attack_type": attack_type,
            "confidence": float(p_attack)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM request_logs")
    total_requests = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM request_logs WHERE true_label = 'attack'")
    total_attacks = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM request_logs WHERE true_label = 'normal'")
    total_normals = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM request_logs WHERE is_correct = 1")
    correct_predictions = c.fetchone()[0]
    
    accuracy = 0
    if total_requests > 0:
        accuracy = round((correct_predictions / total_requests) * 100, 2)
    
    c.execute("SELECT id, timestamp, true_label, predicted_label, attack_type, is_correct FROM request_logs ORDER BY id DESC LIMIT 20")
    recent_logs = []
    for row in c.fetchall():
        recent_logs.append({
            "id": row[0],
            "timestamp": row[1],
            "true_label": row[2],
            "predicted_label": row[3],
            "attack_type": row[4] if row[4] else '',
            "is_correct": bool(row[5])
        })
    
    c.execute("SELECT attack_type, COUNT(*) FROM request_logs WHERE attack_type != '' AND attack_type IS NOT NULL GROUP BY attack_type")
    attack_type_dist = {}
    for row in c.fetchall():
        attack_type_dist[row[0]] = row[1]
    
    conn.close()
    
    return jsonify({
        "total_requests": total_requests,
        "total_attacks": total_attacks,
        "total_normals": total_normals,
        "accuracy": accuracy,
        "recent_logs": recent_logs,
        "attack_type_distribution": attack_type_dist
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
