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

import re
from difflib import get_close_matches

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
        columns = list(df.columns)
        prompt_cols = ", ".join(columns)
        prompt = (
            "The dataset has the following columns: "
            f"{prompt_cols}. Which one is most likely the target? "
            "Return ONLY the column name."
        )

        response = self.llm.invoke(prompt)

        # Pull candidate words out of the reply
        words = re.findall(r"\w+", response)

        # 1️⃣ exact hit
        target = next((w for w in words if w in columns), None)

        # 2️⃣ fuzzy backup (handles minor typos / different‑case)
        if not target:
            match = get_close_matches(words[0], columns, n=1, cutoff=0.8)
            target = match[0] if match else columns[-1]   # last‑column fallback

        task_type = "classification" if df[target].nunique() <= 10 else "regression"
        message   = f"I suggest using `{target}` as the target for a **{task_type}** task."

        return message, target, task_type

