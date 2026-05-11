import warnings
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from train import load_and_prepare_binary, RANDOM_STATE

warnings.filterwarnings('ignore')


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, scaler, feat = load_and_prepare_binary()

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()

    print(f"\n{'=' * 40}")
    print(f"  RANDOM FOREST RESULTS")
    print(f"{'=' * 40}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  TPR (attack detected): {tp/(tp+fn):.4f}  ({tp}/{tp+fn})")
    print(f"  TNR (normal correct):  {tn/(tn+fp):.4f}  ({tn}/{tn+fp})")
    print(f"  False alarms: {fp}/{tn+fp} ({fp/(tn+fp)*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'attack']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf.pkl')
    print("\nSaved models/rf.pkl")


if __name__ == "__main__":
    main()
