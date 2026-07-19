"""Aynı sınıflandırma probleminde dört klasik ML modelini karşılaştırır."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42
    )
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        print(name, {
            "F1": round(f1_score(y_test, probability >= .5), 4),
            "ROC_AUC": round(roc_auc_score(y_test, probability), 4),
        })
