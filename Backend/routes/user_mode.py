from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
from transformers import pipeline # type: ignore
from huggingface_hub import login
import os
from dotenv import load_dotenv  # <-- Add this import

router = APIRouter()

# Load environment variables from .env file
load_dotenv("environment.env")  # <-- Load your env file

# Call login once at module load
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(hf_token)
else:
    raise RuntimeError("HuggingFace token not found in environment variable HF_TOKEN")

# Singleton pattern for model loading to avoid reloading on every request
_sentence_model = None
_gen_pipeline = None  # Add this for the text2text-generation pipeline

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def get_gen_pipeline():
    global _gen_pipeline
    if _gen_pipeline is None:
        _gen_pipeline = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-248M", temperature=0)
    return _gen_pipeline

def extract_text_from_pdf(file):
    """Extracts all text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def compute_similarity(text1, text2):
    """Computes cosine similarity between two texts using sentence embeddings."""
    model = get_sentence_model()
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    score = max(0, min(100, int(sim * 100)))
    return score

@router.post("/analyze")
async def analyze(jd_pdf: UploadFile = File(...), resume_pdf: UploadFile = File(...)):
    # Basic similarity score between JD and resume
    jd_text = extract_text_from_pdf(jd_pdf.file)
    resume_text = extract_text_from_pdf(resume_pdf.file)
    print(f"JD Text: {jd_text}...")
    print(f"Resume Text: {resume_text}...")
    score = compute_similarity(jd_text, resume_text)
    return JSONResponse({"score": score})

def get_match_level(score):
    if score >= 75:
        return "strong match"
    elif score >= 50:
        return "almost a match"
    else:
        return "not a match"

@router.post("/insights")
async def insights(jd_pdf: UploadFile = File(...), resume_pdf: UploadFile = File(...)):
    # Extract text from PDFs
    jd_text = extract_text_from_pdf(jd_pdf.file)
    resume_text = extract_text_from_pdf(resume_pdf.file)
    score = compute_similarity(jd_text, resume_text)
    match_level = get_match_level(score)
    # Use the singleton pipeline
    gen = get_gen_pipeline()
 
    prompt = f"""
    You are a resume reviewer.
    Based on the candidate's resume and the job description, write a short 2-3 sentence feedback (max 60 words) for the candidate.
    Clearly say: this is a {match_level}. If skills are missing, mention up to 2 key ones that should be improved.
    Job Description:
    {jd_text}
    Resume:
    {resume_text}
    Feedback:
    """
    # Generate feedback
    output = gen(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"] # type: ignore
    return JSONResponse({"score":score,"feedback": output})