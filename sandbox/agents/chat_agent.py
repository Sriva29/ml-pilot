'''
Chat Agent Class

Expected flow:
[User Input]
   |
   v
[ChatAgent.analyze()]
   |
   v
[LLM or rule-based extractor]
   |
   v
[Parse for key elements]
   ├── Task Type → "regression" / "classification"
   ├── Target Variable → e.g. "price", "churn"
   └── Feature Hints → e.g. "location", "age", "subscription_type"
   |
   v
[Return Project Goal Dict]
   {
     "task": "regression",
     "target": "price",
     "features": ["location", "size", "bedrooms"]
   }
'''

import re
import requests
import subprocess
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from utils.data_helpers import guess_target_column, detect_task_type


class ChatAgent:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="llama2"):
        self.endpoint = endpoint
        self.model = model
        self.ensure_llama_running()

    def ensure_llama_running(self):
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
            if self.model not in result.stdout:
                st.warning(f"Starting {self.model} with ollama...")
                subprocess.Popen(["ollama", "run", self.model])
            else:
                st.info(f"{self.model} already running.")
        except FileNotFoundError:
            st.error("Ollama is not installed or not in PATH.")

    def inspect_dataset(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        message = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        message += "Here are some column names: " + ", ".join(df.columns[:5]) + "...\n"

        target = guess_target_column(df)
        task = detect_task_type(df, target)

        message += f"\nI suggest using '{target}' as the target column.\n"
        message += f"Based on the values in that column, this looks like a **{task}** task.\n"
        message += "Shall we begin with preprocessing?"

        return message, target, task


    def converse(self, conversation_history: List[Dict]) -> Tuple[str, Dict]:
        system_prompt = (
            "You are a friendly AI tutor helping a beginner plan a machine learning project.\n"
            "Your goal is to understand three things: the ML task type (classification or regression), the target variable, and relevant features.\n"
            "Ask one follow-up question at a time if anything is unclear.\n"
            "Only once you clearly know all three, say: 'Great, we’re ready to proceed. Please upload your dataset.'\n"
            "Keep your tone conversational and supportive."
        )

        full_prompt = system_prompt + "\n\n"
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "Assistant:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False
        }

        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        assistant_reply = response.json().get("response", "")

        st.chat_message("assistant").write(assistant_reply)

        goal = {"task": None, "target": None, "features": []}

        if "upload your dataset" in assistant_reply.lower() or "we're ready" in assistant_reply.lower():
            task = "classification" if "classification" in assistant_reply.lower() else ("regression" if "regression" in assistant_reply.lower() else None)
            target_match = re.search(r"target.*?:\s*(\w+)", assistant_reply, re.IGNORECASE)
            target = target_match.group(1) if target_match else None
            features = re.findall(r"['\"](.*?)['\"]", assistant_reply)

            goal = {
                "task": task,
                "target": target,
                "features": features
            }

        return assistant_reply, goal

    def is_complete(self, goal: Dict) -> bool:
        return bool(goal.get("task") and goal.get("target") and goal.get("features"))

    def recommend_dataset(self, project_goal: Dict) -> List[str]:
        task = project_goal.get("task")
        target = project_goal.get("target")
        if task == "regression":
            return ["Boston Housing Dataset", "California Housing", "Kaggle Real Estate"]
        else:
            return ["Titanic Survival", "Customer Churn", "Iris Classification"]

    def summarize_results(self, eval_report: Dict, model: any) -> str:
        summary = f"Your {model.__class__.__name__} model achieved the following:\n"
        for metric, value in eval_report.items():
            summary += f"- {metric}: {value}\n"
        summary += "\nLooks like you're well on your way! 🎯"
        return summary
