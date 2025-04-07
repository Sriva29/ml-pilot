# app/services/llm_service.py
import requests
import hashlib
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(context_hash):
    # This function will retrieve cached responses
    # Implementation details depend on your storage solution
    pass

def get_llm_response(context):
    # Create a hash of the context for caching
    context_hash = hashlib.md5(context.encode()).hexdigest()
    
    # Check cache first
    cached = get_cached_response(context_hash)
    if cached:
        return cached
    
    # Call LLM API (OpenAI, Ollama, etc.)
    # For the prototype, you could use a simpler model or even mock responses
    response = requests.post(
        "http://localhost:11434/api/generate",  # Example for local Ollama
        json={"model": "mistral", "prompt": context}
    ).json()["response"]
    
    # Cache the response
    cache_response(context_hash, response)
    
    return response