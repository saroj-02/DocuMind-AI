import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")


@st.cache_resource
def get_client():
    """
    Create one reusable Gemini client for this Streamlit process.
    Do not call client.close() after individual requests; doing so can cause
    'Cannot send a request, as the client has been closed' on later reruns.
    """
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add your API key to the .env file "
            "and restart Streamlit."
        )
    return genai.Client(api_key=API_KEY)
