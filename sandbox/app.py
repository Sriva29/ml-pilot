import streamlit as st
import pandas as pd
from agents.chat_agent import ChatAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_training_agent import ModelTrainerAgent
from agents.hyperparameter_agent import HyperparameterTunerAgent
from agents.evaluation_agent import EvaluationAgent
from utils.data_helpers import detect_task_type
from utils.export_helpers import generate_pdf_summary
import pickle



st.set_page_config(page_title="Data Science Assistant", layout="centered")
st.title("🤖 Multi-Agent Data Science Assistant")

# --- Initialize agents ---
chat_agent = ChatAgent()
preprocessing_agent = PreprocessingAgent()
model_trainer = ModelTrainerAgent()
tuner_agent = HyperparameterTunerAgent()
evaluation_agent = EvaluationAgent()

# --- Initialize session state ---
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "project_goal" not in st.session_state:
    st.session_state["project_goal"] = {}
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "preprocessed" not in st.session_state:
    st.session_state["preprocessed"] = False

# --- Upload dataset ---
st.write("### Step 1: Upload your dataset (CSV only)")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file and st.session_state["dataset"] is None:
    df = pd.read_csv(uploaded_file)
    st.session_state["dataset"] = df
    st.success("✅ Dataset uploaded!")

    message, target, task = chat_agent.inspect_dataset(df)
    st.session_state["conversation_history"].append({"role": "assistant", "content": message})
    st.session_state["suggested_target"] = target
    st.session_state["suggested_task"] = task
    st.session_state["show_suggestion"] = True
    st.session_state["confirmed_target_task"] = False

# --- Suggest target column and task ---
if st.session_state.get("show_suggestion") and not st.session_state.get("confirmed_target_task"):
    st.markdown(f"""
    ### My Suggestion:
    - **Target column:** `{st.session_state['suggested_target']}`
    - **Task type:** `{st.session_state['suggested_task']}`
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, let's proceed"):
            st.session_state["project_goal"] = {
                "target": st.session_state["suggested_target"],
                "task": st.session_state["suggested_task"],
                "features": []
            }
            st.session_state["confirmed_target_task"] = True
    with col2:
        if st.button("❌ No, choose another target"):
            st.session_state["manual_target_select"] = True

if st.session_state.get("manual_target_select") and not st.session_state.get("confirmed_target_task"):
    st.markdown("### Choose your own target column:")
    selected = st.selectbox("Select target column", st.session_state["dataset"].columns)
    if st.button("Confirm Target Column"):
        custom_target = selected
        custom_task = detect_task_type(st.session_state["dataset"], custom_target)
        st.session_state["project_goal"] = {
            "target": custom_target,
            "task": custom_task,
            "features": []
        }
        st.session_state["confirmed_target_task"] = True
        st.success(f"✅ You selected '{custom_target}' as target for a {custom_task} task.")

# --- Preprocessing ---
df = st.session_state["dataset"]
goal = st.session_state["project_goal"]

if df is not None and st.session_state.get("confirmed_target_task") and not st.session_state.get("preprocessed"):
    target_col = goal["target"]
    summary = preprocessing_agent.generate_summary(df, target_col)
    st.markdown(summary)

    custom_mode = st.radio("Choose how to apply preprocessing:", ["Full Auto", "Custom"])

    apply_missing = True
    apply_encoding = True
    apply_scaling = True

    if custom_mode == "Custom":
        steps = preprocessing_agent.suggest_preprocessing_steps(df)
        apply_missing = any("Impute" in s for s in steps)
        apply_encoding = any("One-hot" in s for s in steps)
        apply_scaling = any("Scale" in s for s in steps)
        st.write("Select which preprocessing steps to apply:")
        apply_missing = st.checkbox("Handle missing values", value=apply_missing)
        apply_encoding = st.checkbox("Encode categorical variables", value=apply_encoding)
        apply_scaling = st.checkbox("Scale numeric features", value=apply_scaling)

    if st.button("Apply Preprocessing Steps"):
        X, y = preprocessing_agent.preprocess(
            df,
            target_column=target_col,
            apply_missing=apply_missing,
            apply_encoding=apply_encoding,
            apply_scaling=apply_scaling
        )
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["preprocessed"] = True
        st.success("✅ Preprocessing complete!")
        st.write("### Sample of Preprocessed Features")
        st.dataframe(X.head())
        st.write("### Sample of Target Values")
        st.write(y.head())

# --- Model Training ---
if st.session_state.get("preprocessed") and not st.session_state.get("model_trained"):
    task = st.session_state["project_goal"]["task"]
    available_models = model_trainer.suggest_models(task)
    st.write("## Model Selection & Training")
    selected_model = st.selectbox("Available Models", available_models)
    st.session_state["selected_model_name"] = selected_model

    if st.button("Train Model"):
        with st.spinner("Training your model..."):
            model, metrics = model_trainer.train_and_evaluate(
                st.session_state["X"],
                st.session_state["y"],
                selected_model,
                task
            )
            st.session_state["trained_model"] = model
            st.session_state["model_metrics"] = metrics
            st.session_state["model_trained"] = True
            st.success("✅ Model training complete!")

            results, visuals = evaluation_agent.evaluate(
                model,
                {"X": st.session_state["X"], "y": st.session_state["y"]},
                st.session_state["project_goal"]
            )
            st.session_state["evaluation_results"] = results
            st.session_state["evaluation_plots"] = visuals

# --- Evaluation Results ---
if st.session_state.get("model_trained"):
    st.markdown(f"**Model:** {st.session_state['selected_model_name']}")
    st.markdown(model_trainer.summarize_results(
        st.session_state["model_metrics"],
        st.session_state["trained_model"]
    ))

# --- Evaluation Plots ---
if st.session_state.get("evaluation_results"):
    st.markdown("### 📈 Detailed Evaluation")
    for metric, value in st.session_state["evaluation_results"].items():
        if isinstance(value, float):
            st.write(f"**{metric}:** {value:.4f}")
        else:
            st.write(f"**{metric}:** {value}")

if st.session_state.get("evaluation_plots"):
    for title, base64_image in st.session_state["evaluation_plots"].items():
        st.markdown(f"#### {title.replace('_', ' ').title()}")
        st.image(base64_image, use_column_width=True)

# --- Hyperparameter Tuning ---
if st.session_state.get("model_trained"):
    st.markdown("### Want to try tuning your model for better performance?")
    if st.button("🔧 Tune Hyperparameters"):
        model_name = st.session_state["selected_model_name"]
        with st.spinner("Consulting the dark grid..."):
            tuned_model, tuned_metrics = tuner_agent.tune(
                st.session_state["X"],
                st.session_state["y"],
                model_name,
                st.session_state["project_goal"]["task"]
            )
            st.session_state["tuned_model"] = tuned_model
            st.session_state["tuned_metrics"] = tuned_metrics
            st.session_state["model_tuned"] = True
            st.success("✨ Tuning complete. Behold the improved metrics.")

# --- Tuned Results ---
if st.session_state.get("model_tuned"):
    st.markdown("### Tuned Model Evaluation")
    for key, value in st.session_state["tuned_metrics"].items():
        if key == "Best Params":
            st.write(f"**Best Parameters:** `{value}`")
        else:
            st.write(f"**{key}:** {value:.4f}")
    # Re-evaluate the tuned model
    results, visuals = evaluation_agent.evaluate(
        st.session_state["tuned_model"],
        {"X": st.session_state["X"], "y": st.session_state["y"]},
        st.session_state["project_goal"]
    )
    st.session_state["tuned_evaluation_results"] = results
    st.session_state["tuned_evaluation_plots"] = visuals
    if st.session_state.get("tuned_evaluation_results"):
        st.markdown("### 📈 Detailed Evaluation (Tuned Model)")
        for metric, value in st.session_state["tuned_evaluation_results"].items():
            if isinstance(value, float):
                st.write(f"**{metric}:** {value:.4f}")
            else:
                st.write(f"**{metric}:** {value}")

    if st.session_state.get("tuned_evaluation_plots"):
        for title, base64_image in st.session_state["tuned_evaluation_plots"].items():
            st.markdown(f"#### {title.replace('_', ' ').title()}")
            st.image(base64_image, use_column_width=True)
    
if st.session_state.get("model_tuned"):
    import pickle
    from utils.export_helpers import generate_pdf_summary

    # Save tuned model
    with open("tuned_model.pkl", "wb") as f:
        pickle.dump(st.session_state["tuned_model"], f)

    st.markdown("### 📥 Download Your Artifacts")

    col1, col2 = st.columns(2)

    with col1:
        with open("tuned_model.pkl", "rb") as f:
            st.download_button(
                "💾 Download Tuned Model (.pkl)",
                f,
                file_name="tuned_model.pkl",
                mime="application/octet-stream"
            )

    with col2:
        if st.button("🧾 Generate PDF Summary"):
            path = generate_pdf_summary(
                st.session_state["project_goal"],
                st.session_state["selected_model_name"],
                st.session_state["model_metrics"],
                st.session_state["tuned_metrics"]
            )
            with open(path, "rb") as f:
                st.download_button(
                    "📄 Download PDF Summary",
                    f,
                    file_name="model_summary.pdf",
                    mime="application/pdf"
                )



