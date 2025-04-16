# AI Co-Pilot
# 🤖 Multi-Agent Data Science Assistant

A conversational, intelligent assistant for machine learning projects — designed to guide learners and analysts from dataset upload to model training, tuning, and evaluation using a suite of collaborative AI agents.

---

## 🌟 Features

- 💬 **Conversational Interface**: Clarifies project goals using a local LLM (via Ollama)
- 📂 **Data Upload + Inspection**: Accepts CSV files, previews structure
- 🎯 **Goal Inference**: Suggests target column and task type (classification/regression)
- 🧼 **Preprocessing Agent**: Applies imputation, encoding, and scaling (custom or auto)
- 🤖 **Model Training Agent**: Selects and trains ML models
- 🔧 **Hyperparameter Tuning Agent**: Refines models via GridSearchCV
- 📈 **Evaluation Agent**: Generates confusion matrix, ROC, AUC, F1, RMSE, R²
- 📥 **Export**: Download trained models (`.pkl`) and summary reports (`.pdf`)

---

## 🛠️ Setup Instructions

### 1. Clone This Repository

```bash
git clone https://github.com/your-username/multi-agent-ds-assistant.git
cd multi-agent-ds-assistant
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
You'll need:

streamlit

scikit-learn

pandas

matplotlib, seaborn

fpdf

ollama + local LLM like llama3

3. Run the App
bash
Copy
Edit
streamlit run app.py
🧠 Make sure ollama is running locally with your model pulled (e.g., ollama pull llama3)

📂 Folder Structure
Copy
Edit
├── agents/
│   ├── chat_agent.py
│   ├── preprocessing_agent.py
│   ├── model_training_agent.py
│   ├── hyperparameter_agent.py
│   └── evaluation_agent.py
├── utils/
│   ├── data_helpers.py
│   └── export_helpers.py
├── app.py
├── README.md
💡 Example Use Case
“I have a dataset of wine qualities and I want to predict quality based on acidity, sugar, and alcohol content.”

This assistant will:

Suggest a classification task with target = quality

Ask if you want full or custom preprocessing

Train a Logistic Regression model (or your choice)

Tune its hyperparameters

Show ROC/AUC/F1

Let you download a .pkl and a .pdf report of your model’s performance

🧪 Ideal For
Instructors and students in ML/Data Science courses

Analysts building pipelines from scratch

Learners needing scaffolding and feedback

🧠 Powered By
Streamlit

scikit-learn

Ollama + LLaMA2 or LLaMA3

FPDF2 for report generation