# app/routers/ml_operations.py
from fastapi import APIRouter, UploadFile, File, Depends
from ..core.mcp import MLMentorContext
from ..services.llm_service import get_llm_response

router = APIRouter(prefix="/ml", tags=["ML Operations"])

@router.post("/analyze_data")
async def analyze_data(file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    # Process uploaded file
    df = pd.read_csv(file.file)
    
    # Get or create user context
    context = get_user_context(user_id)
    
    # Update context with dataset info
    context.sections["ml_state"]["dataset"] = {
        "columns": list(df.columns),
        "shape": df.shape,
        "dtypes": str(df.dtypes)
    }
    
    # Get LLM response using MCP
    prompt = f"The user has uploaded a dataset with {df.shape[0]} rows and {df.shape[1]} columns. Provide an analysis of this dataset and suggest potential machine learning approaches."
    context.add_message("user", prompt)
    
    response = get_llm_response(context.generate_context())
    
    context.add_message("assistant", response)
    save_user_context(user_id, context)
    
    return {"message": response}