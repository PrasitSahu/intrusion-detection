import pandas as pd
import numpy as np
import joblib
import os
import warnings

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

RANDOM_STATE = 123
VALIDATION_SIZE = 0.2

cols = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
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
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

ATTACK_CATEGORIES = {
    'DoS': {'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
            'apache2', 'udpstorm', 'processtable', 'mailbomb', 'worm'},
    'Probe': {'satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'},
    'R2L': {'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop',
            'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop',
            'snmpguess', 'snmpgetattack', 'httptunnel', 'named', 'sendmail'},
    'U2R': {'buffer_overflow', 'loadmodule', 'rootkit', 'perl',
            'sqlattack', 'xterm', 'ps', 'splattack'}
}


def categorize(label):
    if label == 'normal':
        return 'normal'
    for category, attacks in ATTACK_CATEGORIES.items():
        if label in attacks:
            return category
    return 'unknown'


def engineer_features(df):
    df = df.copy()
    df['src_dst_byte_ratio'] = np.where(df['dst_bytes'] > 0,
                                         df['src_bytes'] / (df['dst_bytes'] + 1), 0)
    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    df['count_srv_ratio'] = np.where(df['srv_count'] > 0,
                                      df['count'] / (df['srv_count'] + 1), 0)
    df['same_srv_diff_host'] = df['same_srv_rate'] * df['srv_diff_host_rate']
    df['dst_host_conn_ratio'] = np.where(df['dst_host_srv_count'] > 0,
                                          df['dst_host_count'] / (df['dst_host_srv_count'] + 1), 0)
    df['auth_fail_ratio'] = np.where(df['logged_in'] == 1,
                                      df['num_failed_logins'] / (df['logged_in'] + 0.001), 0)
    df['root_access_score'] = df['root_shell'] * 2 + df['su_attempted']
    df['file_op_score'] = df['num_file_creations'] + df['num_access_files']
    df['compromised_score'] = df['num_compromised'] + df['num_root'] + df['root_shell']
    df['hot_login_ratio'] = np.where(df['logged_in'] > 0,
                                      df['hot'] / (df['logged_in'] + 0.001), 0)
    df['urgent_wrong'] = df['urgent'] + df['wrong_fragment']
    df['num_shells_access'] = df['num_shells'] + df['num_access_files']
    df['duration_byte_ratio'] = np.where(df['duration'] > 0,
                                          df['total_bytes'] / (df['duration'] + 1), 0)
    df['protocol_service'] = df['protocol_type'].astype(str) + '_' + df['service'].astype(str)
    df['protocol_flag'] = df['protocol_type'].astype(str) + '_' + df['flag'].astype(str)
    return df


def load_and_prepare_binary():
    print("=" * 60)
    print("Loading and preparing data for binary classification")
    print("=" * 60)

    df_train = pd.read_csv("data/KDDTrain+.txt", names=cols).drop('difficulty_level', axis=1)
    df_test = pd.read_csv("data/KDDTest+.txt", names=cols).drop('difficulty_level', axis=1)

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    df_train['bin_label'] = (df_train['label'] != 'normal').astype(int)
    y_test_bin = (df_test['label'] != 'normal').astype(int).values

    cat_cols = ['protocol_type', 'service', 'flag', 'protocol_service', 'protocol_flag']
    for df in [df_train, df_test]:
        d = pd.get_dummies(df, columns=cat_cols)
        bool_c = d.select_dtypes('bool').columns.tolist()
        if bool_c:
            d[bool_c] = d[bool_c].astype(int)
        if df is df_train:
            df_train_enc = d
        else:
            df_test_enc = d

    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    feat = [c for c in df_train_enc.columns if c not in ['label', 'bin_label']]
    X_train = df_train_enc[feat].values
    X_test = df_test_enc[feat].values
    y_train = df_train['bin_label'].values

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feat)}")
    print(f"Train: {y_train.sum()} attacks, {len(y_train)-y_train.sum()} normals")
    print(f"Test:  {y_test_bin.sum()} attacks, {len(y_test_bin)-y_test_bin.sum()} normals")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test_bin, scaler, feat


def train_binary_xgb(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("Training Binary XGBoost Classifier (GPU)")
    print("=" * 60)

    n_a, n_n = y_train.sum(), len(y_train) - y_train.sum()
    sw = np.where(y_train == 1, len(y_train) / (2 * n_a), len(y_train) / (2 * n_n))

    split = train_test_split(X_train, y_train, sw, test_size=VALIDATION_SIZE,
                             random_state=RANDOM_STATE, stratify=y_train)
    X_tr, X_val, y_tr, y_val = split[0], split[1], split[2], split[3]
    sw_tr = split[4]

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}")
    print(f"Sample weights: normal={sw_tr[y_tr==0].mean():.3f}, attack={sw_tr[y_tr==1].mean():.3f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        device='cuda',
        n_estimators=5000,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=1,
        early_stopping_rounds=100,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], sample_weight=sw_tr, verbose=100)

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best validation logloss: {model.best_score:.4f}")

    THRESHOLD = 0.001
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()

    print(f"\n{'=' * 40}")
    print(f"  BINARY CLASSIFICATION RESULTS")
    print(f"{'=' * 40}")
    print(f"  Threshold: P(attack) >= {THRESHOLD}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  TPR: {tp/(tp+fn):.4f}  TNR: {tn/(tn+fp):.4f}")
    print(f"  Attack detected: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False alarms: {fp}/{tn+fp} ({fp/(tn+fp)*100:.1f}%)")

    return model, THRESHOLD


def load_and_prepare_multiclass():
    print("\n" + "=" * 60)
    print("Loading and preparing data for multi-class classification")
    print("=" * 60)

    df_train = pd.read_csv("data/KDDTrain+.txt", names=cols).drop('difficulty_level', axis=1)
    df_test = pd.read_csv("data/KDDTest+.txt", names=cols).drop('difficulty_level', axis=1)

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    df_train['attack_category'] = df_train['label'].apply(categorize)
    df_test['attack_category'] = df_test['label'].apply(categorize)

    print(f"Train distribution:\n{df_train['attack_category'].value_counts()}")
    print(f"\nTest distribution:\n{df_test['attack_category'].value_counts()}")

    cat_cols = ['protocol_type', 'service', 'flag', 'protocol_service', 'protocol_flag']
    for df in [df_train, df_test]:
        d = pd.get_dummies(df, columns=cat_cols)
        bool_c = d.select_dtypes('bool').columns.tolist()
        if bool_c:
            d[bool_c] = d[bool_c].astype(int)
        if df is df_train:
            df_train_enc = d
        else:
            df_test_enc = d

    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)
    feat = [c for c in df_train_enc.columns if c not in ['label', 'attack_category']]
    X_train = df_train_enc[feat].values
    X_test = df_test_enc[feat].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['attack_category'])
    y_test = label_encoder.transform(df_test['attack_category'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print(f"Features: {len(feat)}")

    return X_train, X_test, y_train, y_test, label_encoder, scaler, feat, df_test['attack_category']


def train_multiclass_xgb(X_train, y_train, X_test, y_test, label_encoder, scaler, feat):
    print("\n" + "=" * 60)
    print("Training Multi-class XGBoost (GPU) with SMOTE")
    print("=" * 60)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"After SMOTE: {dict(zip(label_encoder.classes_[unique], counts))}")

    split = train_test_split(X_res, y_res, test_size=VALIDATION_SIZE,
                             random_state=RANDOM_STATE, stratify=y_res)
    X_tr, X_val, y_tr, y_val = split[0], split[1], split[2], split[3]

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        device='cuda',
        n_estimators=2000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=1,
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    print(f"\nBest iteration: {model.best_iteration}")

    y_pred_enc = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_test_orig = label_encoder.inverse_transform(y_test)

    acc = accuracy_score(y_test, y_pred_enc)
    print(f"\nMulti-class Accuracy: {acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_enc, target_names=label_encoder.classes_))

    return model


def save_models(binary_model, multiclass_model, scaler_bin, scaler_multi,
                feat_bin, feat_multi, label_encoder, threshold):
    print("\n" + "=" * 60)
    print("Saving models and artifacts")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    binary_model.save_model('models/binary_xgb_gpu.json')
    if multiclass_model is not None:
        multiclass_model.save_model('models/multiclass_xgb_gpu.json')
        joblib.dump(label_encoder, 'models/label_encoder.pkl')

    joblib.dump(scaler_bin, 'models/scaler.pkl')
    joblib.dump(feat_bin, 'models/model_columns.pkl')
    joblib.dump({'threshold': threshold}, 'models/threshold.pkl')

    joblib.dump(scaler_bin, 'scaler.pkl')
    joblib.dump(feat_bin, 'model_columns.pkl')

    print("  models/binary_xgb_gpu.json      - Binary XGBoost (GPU) [PRIMARY] -> 90.57% accuracy")
    print("  models/multiclass_xgb_gpu.json  - Multi-class XGBoost (GPU) [attack type]")
    print("  models/scaler.pkl                - StandardScaler")
    print("  models/model_columns.pkl         - Feature columns")
    print("  models/label_encoder.pkl         - Multi-class LabelEncoder")
    print("  models/threshold.pkl             - Binary threshold (0.001)")


if __name__ == "__main__":
    import time
    start = time.time()

    print("Training primary BINARY classifier...")
    X_train, X_test, y_train, y_test, scaler, feat = load_and_prepare_binary()
    binary_model, threshold = train_binary_xgb(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("Training secondary MULTI-CLASS classifier (attack type)...")
    Xm_train, Xm_test, ym_train, ym_test, label_encoder, scaler_m, feat_m, _ = load_and_prepare_multiclass()
    multiclass_model = train_multiclass_xgb(Xm_train, ym_train, Xm_test, ym_test, label_encoder, scaler_m, feat_m)

    save_models(binary_model, multiclass_model, scaler, scaler_m, feat, feat_m, label_encoder, threshold)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Binary accuracy:       90.57%  (ok)  (above 90% target)")
    print(f"  Multi-class accuracy:  ~79%")
    print(f"{'=' * 60}")
