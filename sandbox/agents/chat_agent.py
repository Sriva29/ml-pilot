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

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



class ChatAgent:
    def __init__(self):
        # 🔐 Auth token already set via Streamlit secrets or environment
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            st.error("❌ Hugging Face API token not found. Please add it to your secrets.")
            raise RuntimeError("Missing Hugging Face API token")

        self.llm = HuggingFaceEndpoint(
            repo_id="tiiuae/falcon-rw-1b",
            task="text-generation",
            temperature=0.7,
            max_new_tokens=256
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

    def inspect_dataset(self, df):
        """Suggest target column and task type"""
        columns = ", ".join(df.columns)
        text = f"The dataset has the following columns: {columns}. What column is most likely the target for machine learning?"
        response = self.llm.invoke(text)

        target = response.strip().split()[0]
        task_type = "classification" if df[target].nunique() <= 10 else "regression"
        message = f"I suggest using `{target}` as the target for a **{task_type}** task."
        return message, target, task_type

