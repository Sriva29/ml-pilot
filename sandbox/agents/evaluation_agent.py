from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)
import pandas as pd
import io
import base64
from sklearn.preprocessing import label_binarize
from math import sqrt

class EvaluationAgent:
    def __init__(self):
        pass

    def evaluate(self, model: Any, data: Dict[str, pd.DataFrame], goal: dict) -> Tuple[Dict[str, Any], Dict[str, str]]:
        X = data["X"]
        y = data["y"]
        task = goal.get("task")

        results = {}
        visuals = {}

        y_pred = model.predict(X)

        if task == "classification":
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
            else:
                y_proba = y_pred

            results["Accuracy"] = accuracy_score(y, y_pred)
            results["F1 Score"] = f1_score(y, y_pred, average="weighted")

            if len(set(y)) > 2:
                y_bin = label_binarize(y, classes=list(set(y)))
                results["AUC"] = roc_auc_score(y_bin, y_proba, multi_class="ovr")
            else:
                y_binary_proba = y_proba[:, 1] if hasattr(model, "predict_proba") and y_proba.ndim > 1 else y_proba
                results["AUC"] = roc_auc_score(y, y_binary_proba)

            visuals["confusion_matrix"] = self._plot_confusion_matrix(y, y_pred)
            roc = self._plot_roc_curve(y, y_proba)
            if roc:
                visuals["roc_curve"] = roc

        else:  # Regression
            results["RMSE"] = sqrt(mean_squared_error(y, y_pred))
            results["R^2 Score"] = r2_score(y, y_pred)

        return results, visuals

    def _plot_confusion_matrix(self, y_true, y_pred) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        return self._fig_to_base64(fig)

    def _plot_roc_curve(self, y_true, y_proba) -> str:
        if len(set(y_true)) > 2:
            return None  # Skip ROC curve for multiclass

        y_binary_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        fpr, tpr, _ = roc_curve(y_true, y_binary_proba)
        roc_auc = roc_auc_score(y_true, y_binary_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="green")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"
