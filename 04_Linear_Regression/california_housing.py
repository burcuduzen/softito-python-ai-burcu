"""California Housing ile Ridge Regression, CV ve regresyon metrikleri."""
from pathlib import Path
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
        "R2": r2_score(y_test, prediction),
        "CV_R2": cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean(),
    }
    print({key: round(value, 4) for key, value in metrics.items()})
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "ridge_model.joblib")
