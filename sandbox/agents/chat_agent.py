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
import os
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict

from utils.data_helpers import guess_target_column, detect_task_type

from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatAgent:
    def __init__(self):
        # 🔐 Auth token already set via Streamlit secrets or environment
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            st.error("❌ Hugging Face API token not found. Please add it to your secrets.")
            raise RuntimeError("Missing Hugging Face API token")

        self.llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-rw-1b",
            model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful data science assistant.
        A user is trying to build a machine learning model.
        Ask clarifying questions if needed, and once enough info is gathered, summarize it.

        Current conversation:
        {history}
        """)
        self.output_parser = StrOutputParser()

    def ask(self, history: str) -> str:
        chain = self.prompt | self.llm | self.output_parser
        return chain.invoke({"history": history})

    def inspect_dataset(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        text = f"You’re a helpful assistant. A user uploaded a dataset with these columns: {', '.join(df.columns)}. Which column is most likely the target for ML? Reply with just the column name and reason."
        response = self.llm.invoke(text).strip()

        pattern = re.compile(r"[`'\"]?(\b(?:%s)\b)[`'\"]?" % "|".join(re.escape(col) for col in df.columns))
        match = pattern.search(response)
        if not match:
            raise ValueError(f"Could not determine target column from LLM response: {response}")

        target = match.group(1)
        task_type = "classification" if df[target].nunique() <= 10 else "regression"
        message = f"I suggest using `{target}` as the target for a **{task_type}** task.\n\nLLM response:\n> {response}"

        return message, target, task_type

