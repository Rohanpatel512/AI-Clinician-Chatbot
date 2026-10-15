from pathlib import Path 
import os 
from dotenv import load_dotenv 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = os.getenv("MODEL_REPO")
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME")
HF_MODEL_TOKEN = os.getenv("HF_MODEL_TOKEN")
MODAL_API_URL = os.getenv("MODAL_API_URL")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4000"
    ).split(",")
    if origin.strip()
]