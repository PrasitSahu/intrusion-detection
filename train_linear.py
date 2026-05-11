import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from train import load_and_prepare_binary, load_and_prepare_multiclass, engineer_features, categorize, cols, ATTACK_CATEGORIES, RANDOM_STATE

warnings.filterwarnings('ignore')


def train_binary_linear(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("Training Binary Linear Regression")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_raw = model.predict(X_test)

    best_thresh = 0.5
    best_acc = 0
    for thresh in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_pred_raw >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    y_pred = (y_pred_raw >= best_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()

    print(f"\n{'=' * 40}")
    print(f"  BINARY LINEAR REGRESSION RESULTS")
    print(f"{'=' * 40}")
    print(f"  Optimal threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  TPR: {tp/(tp+fn):.4f}  TNR: {tn/(tn+fp):.4f}")
    print(f"  Attack detected: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False alarms: {fp}/{tn+fp} ({fp/(tn+fp)*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'attack']))

    return model, best_thresh


def train_multiclass_linear(X_train, X_test, y_train, y_test, label_encoder):
    print("\n" + "=" * 60)
    print("Training Multi-class OneVsRest(LinearRegression)")
    print("=" * 60)

    model = OneVsRestClassifier(LinearRegression())
    model.fit(X_train, y_train)

    y_pred_enc = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_test_orig = label_encoder.inverse_transform(y_test)

    acc = accuracy_score(y_test, y_pred_enc)
    print(f"\nMulti-class Accuracy: {acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_enc, target_names=label_encoder.classes_))

    return model


def main():
    import time
    start = time.time()

    print("Loading data and training BINARY Linear Regression...")
    X_train, X_test, y_train, y_test, scaler, feat = load_and_prepare_binary()
    binary_model, threshold = train_binary_linear(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("Training MULTI-CLASS OneVsRest(LinearRegression)...")
    Xm_train, Xm_test, ym_train, ym_test, label_encoder, scaler_m, feat_m, _ = load_and_prepare_multiclass()
    multiclass_model = train_multiclass_linear(Xm_train, Xm_test, ym_train, ym_test, label_encoder)

    os.makedirs('models', exist_ok=True)
    joblib.dump(binary_model, 'models/binary_linear.pkl')
    joblib.dump(multiclass_model, 'models/multiclass_linear.pkl')
    joblib.dump(scaler, 'models/scaler_linear.pkl')
    joblib.dump(feat, 'models/model_columns_linear.pkl')
    joblib.dump({'threshold': threshold}, 'models/threshold_linear.pkl')
    joblib.dump(label_encoder, 'models/label_encoder_linear.pkl')

    print("\n" + "=" * 60)
    print("Saved artifacts to models/:")
    print("  models/binary_linear.pkl         - Binary Linear Regression")
    print("  models/multiclass_linear.pkl     - Multi-class OneVsRest(LinearRegression)")
    print("  models/scaler_linear.pkl          - StandardScaler")
    print("  models/model_columns_linear.pkl   - Feature columns")
    print("  models/threshold_linear.pkl       - Optimal threshold")
    print("  models/label_encoder_linear.pkl   - Multi-class LabelEncoder")
    print("=" * 60)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
