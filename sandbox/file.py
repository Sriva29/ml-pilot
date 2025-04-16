import os

# Define the folder structure and files with their boilerplate content
structure = {
    "agents": {
        "chat_agent.py": """\
class ChatAgent:
    def __init__(self):
        pass

    def analyze(self, user_input: str) -> dict:
        \"\"\"Extracts project goal and intent from user message.\"\"\"
        pass

    def recommend_dataset(self, project_goal: dict) -> list:
        \"\"\"Suggests datasets based on the project goal.\"\"\"
        pass

    def summarize_results(self, eval_report: dict, model: any) -> str:
        \"\"\"Generates a human-friendly summary of model results.\"\"\"
        pass
""",
        "preprocessing_agent.py": """\
import pandas as pd
from typing import Tuple

class PreprocessingAgent:
    def __init__(self):
        pass

    def process(self, dataset: pd.DataFrame, goal: dict) -> Tuple[pd.DataFrame, str]:
        \"\"\"Cleans, encodes, and prepares the dataset based on project goal.\"\"\"
        pass
""",
        "orchestrator_agent.py": """\
class OrchestratorAgent:
    def __init__(self):
        pass

    def decide_next_step(self, project_state: dict) -> str:
        \"\"\"Decides the next logical step in the pipeline based on current state.\"\"\"
        pass
""",
        "hyperparameter_agent.py": """\
class HyperparameterAgent:
    def __init__(self):
        pass

    def tune(self, data, goal: dict) -> dict:
        \"\"\"Finds optimal hyperparameters for the chosen model type.\"\"\"
        pass
""",
        "model_training_agent.py": """\
from typing import Tuple, Optional

class ModelTrainingAgent:
    def __init__(self):
        pass

    def train(self, data, goal: dict, params: Optional[dict] = None) -> Tuple[any, dict]:
        \"\"\"Trains a model using data and optionally tuned parameters. Returns the trained model and a training report.\"\"\"
        pass
""",
        "evaluation_agent.py": """\
from typing import Tuple

class EvaluationAgent:
    def __init__(self):
        pass

    def evaluate(self, model: any, data, goal: dict) -> Tuple[dict, any]:
        \"\"\"Evaluates the trained model and returns a report along with visualizations.\"\"\"
        pass
""",
        "__init__.py": ""  # empty file to mark package
    },
    "utils": {
        "dataset_handler.py": """\
import pandas as pd
from io import BytesIO

class DatasetHandler:
    def load(self, uploaded_file: BytesIO) -> pd.DataFrame:
        \"\"\"Reads an uploaded CSV/Excel/JSON into a DataFrame.\"\"\"
        pass
""",
        "__init__.py": ""  # empty file
    },
    "prompts": {
        "chat_goal_prompt.txt": "# Prompt template for extracting project goals",
        "summary_prompt.txt": "# Prompt template for summarizing results"
    },
    "data": {},     # folder for datasets (can be empty)
    "models": {},   # folder for storing model files
    ".": {
        "app.py": """\
import streamlit as st
import pandas as pd

# Import agents from the agents folder
from agents.chat_agent import ChatAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.hyperparameter_agent import HyperparameterAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.evaluation_agent import EvaluationAgent

def main():
    st.title("Multi-Agent Data Science Assistant")
    
    # Step 1: Get user input
    user_input = st.text_input("Describe your data science project:")
    
    if user_input:
        chat_agent = ChatAgent()
        project_goal = chat_agent.analyze(user_input)
        st.session_state["project_goal"] = project_goal
        st.write("Project goal extracted:", project_goal)
    
    # Step 2: Dataset upload/selection
    if "dataset" not in st.session_state:
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)")
        if uploaded_file:
            st.session_state["dataset"] = pd.read_csv(uploaded_file)
    
    # Step 3: Preprocessing
    if "dataset" in st.session_state:
        pre_agent = PreprocessingAgent()
        processed_data, prep_report = pre_agent.process(st.session_state["dataset"], st.session_state.get("project_goal", {}))
        st.session_state["processed_data"] = processed_data
        st.write("Preprocessing Report:", prep_report)
    
    # Step 4: Hyperparameter Tuning (optional)
    if st.button("Run Hyperparameter Tuning"):
        hp_agent = HyperparameterAgent()
        tuned_params = hp_agent.tune(st.session_state.get("processed_data"), st.session_state.get("project_goal", {}))
        st.session_state["tuned_params"] = tuned_params
        st.write("Tuned Parameters:", tuned_params)
    
    # Step 5: Training
    if st.button("Train Model"):
        train_agent = ModelTrainingAgent()
        model, train_report = train_agent.train(
            st.session_state.get("processed_data"),
            st.session_state.get("project_goal", {}),
            st.session_state.get("tuned_params")
        )
        st.session_state["model"] = model
        st.write("Training Report:", train_report)
    
    # Step 6: Evaluation
    if st.button("Evaluate Model"):
        eval_agent = EvaluationAgent()
        eval_report, visuals = eval_agent.evaluate(
            st.session_state.get("model"),
            st.session_state.get("processed_data"),
            st.session_state.get("project_goal", {})
        )
        st.write("Evaluation Report:", eval_report)
        st.pyplot(visuals)
    
    # Step 7: Final Summary via Chat Agent
    if "model" in st.session_state:
        summary = chat_agent.summarize_results(eval_report, st.session_state["model"])
        st.write("Summary:", summary)

if __name__ == "__main__":
    main()
""",
        "requirements.txt": """\
streamlit
pandas
scikit-learn
matplotlib
# Add other libraries as needed
"""
    }
}

# Function to create files recursively
def create_structure(base_path, structure_dict):
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            # Create folder
            os.makedirs(path, exist_ok=True)
            # Recursively create files/folders within
            create_structure(path, content)
        else:
            # Create file and write content
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)

# Create files and folders starting from the current directory
create_structure('.', structure)

print("Project scaffolding created.")
