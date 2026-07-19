"""Dengesiz finansal işlemlerde Isolation Forest anomali tespiti."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import RobustScaler

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (3000, 5))
    fraud = rng.normal(3, 1.7, (60, 5))
    X = np.vstack([normal, fraud])
    y = np.r_[np.zeros(len(normal)), np.ones(len(fraud))]
    X = RobustScaler().fit_transform(X)
    model = IsolationForest(
        contamination=y.mean(), n_estimators=300, random_state=42
    ).fit(X)
    anomaly_score = -model.decision_function(X)
    prediction = (model.predict(X) == -1).astype(int)
    print("ROC-AUC:", round(roc_auc_score(y, anomaly_score), 4))
    print(classification_report(y, prediction, digits=4))
