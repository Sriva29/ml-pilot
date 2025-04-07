# app/routers/lti.py
from fastapi import APIRouter, Request, Depends
from pylti1p3.contrib.fastapi import FastApiMessageLaunch, FastApiOIDCLogin

router = APIRouter(prefix="/lti", tags=["LTI"])

@router.post("/launch")
async def launch(request: Request, launch: FastApiMessageLaunch = Depends()):
    if not launch.is_resource_launch():
        raise HTTPException(status_code=400, message="Invalid launch")
    
    # Extract user info from launch
    user_id = launch.get_claim("sub")
    context_id = launch.get_context_claim("id")
    
    # Generate redirect URL for Streamlit frontend with auth token
    redirect_url = f"http://localhost:8501?token={generate_token(user_id)}"
    
    return {"redirect": redirect_url}