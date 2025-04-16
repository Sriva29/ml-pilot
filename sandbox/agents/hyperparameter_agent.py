import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Tuple, Dict, Any

class HyperparameterTunerAgent:
    def __init__(self):
        self.param_grid = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1.0, 10.0]
            },
            "Random Forest Classifier": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            },
            "Linear Regression": {},  # No hyperparameters to tune
            "Random Forest Regressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        }

        self.models = {
            "Logistic Regression": LogisticRegression,
            "Random Forest Classifier": RandomForestClassifier,
            "Linear Regression": LinearRegression,
            "Random Forest Regressor": RandomForestRegressor
        }

    def tune(self, X: pd.DataFrame, y: pd.Series, model_name: str, task: str) -> Tuple[Any, Dict[str, Any]]:
        ModelClass = self.models[model_name]
        params = self.param_grid.get(model_name, {})

        if not params:
            model = ModelClass()
            model.fit(X, y)
            return model, {"Note": "No hyperparameters to tune for this model."}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid = GridSearchCV(ModelClass(), param_grid=params, cv=3)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        if task == "classification":
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average="weighted"),
                "Best Params": grid.best_params_
            }
        else:
            metrics = {
                "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                "R^2 Score": r2_score(y_test, y_pred),
                "Best Params": grid.best_params_
            }

        return best_model, metrics
