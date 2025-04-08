from fastapi import FastAPI, Depends, Form, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, ChatMessage, init_db
from datetime import datetime
import requests
from fastapi.responses import RedirectResponse
#from pylti.decorators import lti
#from pylti.common import LTIException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.middleware import Middleware



# Initialize database
init_db()

#LTI credentials
CONSUMER_KEY = "mlassistant-key"
SHARED_SECRET = "mlassistant-secret"

class DisableFrameOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers['X-Frame-Options'] = 'ALLOWALL'
        return response

app = FastAPI(
    middleware=[
        Middleware(DisableFrameOptionsMiddleware),
    ]
)


MAX_HISTORY_LENGTH = 20

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatInput(BaseModel):
    session_id: str
    message: str




@app.middleware("http")
async def print_response_headers(request: Request, call_next):
    response = await call_next(request)
    print("🚨 Response headers:", response.headers)
    return response


@app.post("/chat")
def chat_with_db(chat_input: ChatInput, db: Session = Depends(get_db)):
    # 1. Add user message to DB
    user_msg = ChatMessage(
        session_id=chat_input.session_id,
        role="user",
        content=chat_input.message,
        timestamp=datetime.utcnow()
    )
    db.add(user_msg)
    db.commit()

    # 2. Retrieve the last N messages for this session
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_input.session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(MAX_HISTORY_LENGTH)
        .all()
    )
    messages.reverse()  # oldest to newest
#making changes
    # 3. Build the prompt
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"

    # 4. Send to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    reply = response.json()["response"]

    # 5. Save assistant's reply
    bot_msg = ChatMessage(
        session_id=chat_input.session_id,
        role="assistant",
        content=reply,
        timestamp=datetime.utcnow()
    )
    db.add(bot_msg)
    db.commit()

    return {"reply": reply}


@app.get("/lti-launch")
async def preview_launch():
    return HTMLResponse("""
        <h3>✅ ML Assistant Tool Installed</h3>
        <p>This tool is ready for launch via LTI.</p>
        <p>Return to your Moodle course and launch it properly.</p>
    """)

# from fastapi.responses import HTMLResponse
#installed multipart
@app.post("/lti-launch")
async def lti_launch(
    request: Request,
    oauth_consumer_key: str = Form(...),
    lis_person_sourcedid: str = Form(None),
    lis_person_name_full: str = Form(None),
):
    # For now, skip full OAuth validation (safe in local dev)
    learner_id = lis_person_sourcedid or lis_person_name_full or "unknown_user"
    streamlit_url = f"http://localhost:8501/?learner_id={learner_id}"

    html_content = f"""
    <html>
        <body>
            <h2>✅ LTI Launch Received!</h2>
            <p>Welcome, <strong>{learner_id}</strong>!</p>
            <p><a href="{streamlit_url}" target="_blank">👉 Click here to open the ML Assistant</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)



