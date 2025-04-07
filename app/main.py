# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import ml_operations, lti

app = FastAPI(title="ML Mentor")
app.include_router(ml_operations.router)
app.include_router(lti.router)

# Add CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])