from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Step 2: Define the structure of the message we expect
class ChatMessage(BaseModel):
    message: str

# Step 1: Set up the route to receive chat messages
@app.post("/chat")
def chat_with_bot(user_message: ChatMessage):
    # For now, just echo it back
    return {"reply": f"You said: {user_message.message}"}
