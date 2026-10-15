from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import chat_router 
from app.core.config import CORS_ORIGINS

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat_router)