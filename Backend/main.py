from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from routes import user_mode, recruiter_mode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_mode.router, prefix="/user")
app.include_router(recruiter_mode.router, prefix="/recruiter")
