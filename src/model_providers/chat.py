"""
Chat/LLM providers for generating responses.

This is a minimal implementation to maintain compatibility.
For production use, integrate with the new package structure.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_chat_fn():
    """Get chat function based on provider."""
    provider = os.getenv("PROVIDER", "gemini").lower()
    
    if provider == "gemini":
        return _get_gemini_chat()
    elif provider == "openai":
        return _get_openai_chat()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _get_gemini_chat():
    """Get Gemini chat function."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        def chat(messages, temperature=0.2):
            # Simple implementation - just use the last user message
            user_content = messages[-1]["content"] if messages else ""
            response = model.generate_content(user_content)
            return response.text
        
        return chat
        
    except ImportError:
        raise RuntimeError("google-generativeai package not installed")

def _get_openai_chat():
    """Get OpenAI chat function."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        client = OpenAI(api_key=api_key)
        
        def chat(messages, temperature=0.2):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        return chat
        
    except ImportError:
        raise RuntimeError("openai package not installed")
