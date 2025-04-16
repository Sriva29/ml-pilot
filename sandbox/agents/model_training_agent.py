import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Tuple, Dict, Any
import numpy as np

class ModelTrainerAgent:
    def __init__(self):
        self.available_models = {
            "classification": {
                "Logistic Regression": LogisticRegression,
                "Random Forest Classifier": RandomForestClassifier
            },
            "regression": {
                "Linear Regression": LinearRegression,
                "Random Forest Regressor": RandomForestRegressor
            }
        }

    def suggest_models(self, task: str) -> list:
        return list(self.available_models.get(task, {}).keys())

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, model_name: str, task: str) -> Tuple[Any, Dict[str, float]]:
        ModelClass = self.available_models[task][model_name]
        model = ModelClass()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average="weighted")
            }
        else:
            metrics = {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R^2 Score": r2_score(y_test, y_pred)
            }

        return model, metrics

    def summarize_results(self, metrics: Dict[str, float], model: Any) -> str:
        summary = f"Your {model.__class__.__name__} model achieved the following:\n"
        for metric, value in metrics.items():
            summary += f"- {metric}: {value:.4f}\n"
        return summary
