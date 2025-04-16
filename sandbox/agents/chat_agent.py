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


from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load token from secrets or environment (already set in app.py)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

if not HUGGINGFACE_TOKEN:
    st.error("❌ Hugging Face API token not found. Please enter it in the app.")
    raise RuntimeError("Missing Hugging Face API token")

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

class ChatAgent:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful data science assistant.
        A user is trying to build a machine learning model.
        Ask clarifying questions if needed, and once enough info is gathered, summarize it.

        Current conversation:
        {history}
        """)
        self.output_parser = StrOutputParser()

    def ask(self, history):
        chain = self.prompt | llm | self.output_parser
        return chain.invoke({"history": history})

    def inspect_dataset(self, df):
        """Suggest target column and task type"""
        columns = ", ".join(df.columns)
        text = f"The dataset has the following columns: {columns}. What column is most likely the target for machine learning?"
        response = llm.invoke(text)

        target = response.strip().split()[0]
        task_type = "classification" if df[target].nunique() <= 10 else "regression"
        message = f"I suggest using `{target}` as the target for a **{task_type}** task."
        return message, target, task_type



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
