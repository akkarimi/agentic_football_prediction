import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str, default: str = "") -> str:
    """Get value from env vars, fall back to Streamlit secrets when deployed."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

FOOTBALL_DATA_API_KEY = _get("FOOTBALL_DATA_API_KEY")
GROQ_API_KEY          = _get("FTgroqkey")
DB_PATH               = _get("DB_PATH", "predictions.db")
GROQ_MODEL            = _get("GROQ_MODEL", "llama-3.3-70b-versatile")
