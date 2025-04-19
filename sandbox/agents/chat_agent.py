import os
import re
from typing import List
import pandas as pd
import streamlit as st
from difflib import get_close_matches
from pydantic import BaseModel, ValidationError

from utils.data_helpers import detect_task_type

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


class DatasetInfo(BaseModel):
    target: str
    task: str
    features: List[str]


class ChatAgent:
    def __init__(self):
        # Ensure API token is available
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            st.error("❌ Hugging Face API token not found. Please add it to your secrets.")
            raise RuntimeError("Missing Hugging Face API token")

        # Initialize LLM endpoint
        self.llm = HuggingFaceEndpoint(
            repo_id="tiiuae/falcon-rw-1b",
            task="text-generation",
            temperature=0.7,
            max_new_tokens=256
        )

        # Prompt template for free-form chat
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are a helpful data science assistant.
            A user is trying to build a machine learning model.

            Current conversation:
            {history}
            """
        )
        self.free_parse = StrOutputParser()

    def ask(self, history: str) -> str:
        """Handle general chat with the user."""
        chain = self.prompt_template | self.llm | self.free_parse
        return chain.invoke({"history": history})

    def inspect_dataset(self, df: pd.DataFrame) -> DatasetInfo:
        """
        Analyze DataFrame columns and use the LLM to suggest:
          - target variable
          - task type (classification/regression)
          - three feature hints

        Returns a DatasetInfo object.
        """
        # 1️⃣ Normalize column names to lowercase to reduce mismatch issues
        original_columns = list(df.columns)
        normalized_columns = [c.lower() for c in original_columns]
        norm_to_orig = dict(zip(normalized_columns, original_columns))

        # 2️⃣ Use PydanticOutputParser for schema enforcement
        parser = PydanticOutputParser(pydantic_object=DatasetInfo)
        prompt = (
            f"Columns: {normalized_columns}\n"
            "Return a JSON object matching this schema:\n"
            f"{parser.get_format_instructions()}"
        )
        raw = self.llm.invoke(prompt)

        # 3️⃣ Parse into DatasetInfo, fallback on validation error
        try:
            info = parser.parse(raw)
        except ValidationError as ve:
            st.error(f"LLM response malformed: {ve}")
            return self._fallback_inspect(df, original_columns)

        # 4️⃣ Map normalized names back to original casing
        info.target = norm_to_orig.get(info.target.lower(), info.target)
        info.features = [norm_to_orig.get(f.lower(), f) for f in info.features]

        # 5️⃣ Override task based on actual data distribution
        info.task = detect_task_type(df[info.target])

        return info

    def _fallback_inspect(self, df: pd.DataFrame, columns: List[str]) -> DatasetInfo:
        """
        Simple fallback if structured parsing fails: only extract target,
        then infer task and provide empty feature hints.
        """
        prompt_cols = ", ".join(columns)
        prompt = (
            f"The dataset has the following columns: {prompt_cols}. "
            "Return ONLY the column name most likely the target."
        )
        response = self.llm.invoke(prompt)

        # Regex and fuzzy matching to find a valid column
        words = re.findall(r"\w+", response)
        target = next((w for w in words if w in columns), None)
        if not target:
            match = get_close_matches(words[0], columns, n=1, cutoff=0.8)
            target = match[0] if match else columns[-1]

        task = detect_task_type(df[target])
        return DatasetInfo(target=target, task=task, features=[])
