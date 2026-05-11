import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from train import engineer_features, cols, categorize, ATTACK_CATEGORIES


def load_models():
    binary_model = xgb.XGBClassifier()
    binary_model.load_model('models/binary_xgb_gpu.json')

    artifacts = joblib.load('models/threshold.pkl')
    threshold = artifacts['threshold'] if isinstance(artifacts, dict) else artifacts
    scaler = joblib.load('models/scaler.pkl')
    feat = joblib.load('models/model_columns.pkl')

    multiclass_model = None
    label_encoder = None
    try:
        multiclass_model = xgb.XGBClassifier()
        multiclass_model.load_model('models/multiclass_xgb_gpu.json')
        label_encoder = joblib.load('models/label_encoder.pkl')
    except Exception:
        pass

    return binary_model, threshold, scaler, feat, multiclass_model, label_encoder


def preprocess_packet(packet_dict, scaler, feat):
    df = pd.DataFrame([packet_dict])
    df = engineer_features(df)

    cat_cols = ['protocol_type', 'service', 'flag', 'protocol_service', 'protocol_flag']
    df = pd.get_dummies(df, columns=cat_cols)
    bool_c = df.select_dtypes('bool').columns.tolist()
    if bool_c:
        df[bool_c] = df[bool_c].astype(int)
    df = df.reindex(columns=feat, fill_value=0)

    X = scaler.transform(df.values)
    return X


def predict(packet_dict):
    binary_model, threshold, scaler, feat, multiclass_model, label_encoder = load_models()

    X = preprocess_packet(packet_dict, scaler, feat)

    p_attack = binary_model.predict_proba(X)[0, 1]
    is_attack = p_attack >= threshold

    if not is_attack:
        return {'prediction': 'normal', 'confidence': float(1 - p_attack)}
    else:
        if multiclass_model is not None and label_encoder is not None:
            attack_probs = multiclass_model.predict_proba(X)[0]
            attack_type_idx = np.argmax(attack_probs)
            attack_type = label_encoder.classes_[attack_type_idx]
            return {
                'prediction': 'attack',
                'attack_type': str(attack_type),
                'confidence': float(p_attack),
            }
        else:
            return {'prediction': 'attack', 'confidence': float(p_attack)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        print("Evaluating on full test set...")
        df_test = pd.read_csv("data/KDDTest+.txt", names=cols)
        df_test = df_test.drop('difficulty_level', axis=1)

        binary_model, threshold, scaler, feat, multiclass_model, label_encoder = load_models()

        df_test = engineer_features(df_test)
        y_test_orig = df_test['label'].apply(
            lambda x: 'normal' if x == 'normal' else 'attack'
        )

        cat_cols = ['protocol_type', 'service', 'flag', 'protocol_service', 'protocol_flag']
        df_test_enc = pd.get_dummies(df_test, columns=cat_cols)
        bool_c = df_test_enc.select_dtypes('bool').columns.tolist()
        if bool_c:
            df_test_enc[bool_c] = df_test_enc[bool_c].astype(int)
        df_test_enc = df_test_enc.reindex(columns=feat, fill_value=0)

        X_test = scaler.transform(df_test_enc.values)
        p = binary_model.predict_proba(X_test)[:, 1]
        y_pred = (p >= threshold).astype(int)

        y_true = (y_test_orig == 'attack').astype(int)

        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, target_names=['normal', 'attack']))
    else:
        sample = {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
            'flag': 'SF', 'src_bytes': 200, 'dst_bytes': 1000,
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 1,
            'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
            'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
            'num_access_files': 0, 'num_outbound_cmds': 0,
            'is_host_login': 0, 'is_guest_login': 0,
            'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
            'dst_host_count': 1, 'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 0.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
        }
        result = predict(sample)
        print(f"Prediction: {result}")
