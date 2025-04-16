from fpdf import FPDF

def generate_pdf_summary(project_goal, model_name, metrics, tuned_metrics=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Data Science Model Summary", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Task: {project_goal['task']}", ln=True)
    pdf.cell(200, 10, txt=f"Target Column: {project_goal['target']}", ln=True)
    pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt="Metrics:", ln=True)
    for k, v in metrics.items():
        pdf.cell(200, 8, txt=f"{k}: {v}", ln=True)

    if tuned_metrics:
        pdf.ln(5)
        pdf.cell(200, 10, txt="Tuned Model Metrics:", ln=True)
        for k, v in tuned_metrics.items():
            pdf.cell(200, 8, txt=f"{k}: {v}", ln=True)

    output_path = "model_summary.pdf"
    pdf.output(output_path)
    return output_path

