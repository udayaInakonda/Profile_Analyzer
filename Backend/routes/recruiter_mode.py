import faiss
import pickle
from fastapi import APIRouter, File, UploadFile, Body
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
import mysql.connector
import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pydantic import BaseModel

router = APIRouter()
model = SentenceTransformer('all-MiniLM-L6-v2')

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Amzur@#123',
    'database': 'profile_analyzer'
}

# Define the absolute path to the FAISS index and metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "Database", "resume_index.faiss")
FAISS_META_PATH = os.path.join(BASE_DIR, "Database", "resume_metadata.pkl")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def compute_similarity(text1, text2):
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    score = max(0, min(100, int(sim * 100)))
    return score

@router.post("/analyze_bulk")
async def analyze_bulk(
    jd_pdf: UploadFile = File(...),
    resumes: list[UploadFile] = File(...)
):
    jd_text = extract_text_from_pdf(jd_pdf.file)
    results = []
    for resume in resumes:
        resume_text = extract_text_from_pdf(resume.file)
        score = compute_similarity(jd_text, resume_text)
        results.append({"resume_name": resume.filename, "score": score})
    return JSONResponse({"results": results})


@router.post("/analyze_db_bulk")
async def analyze_db_bulk(jd_pdf: UploadFile = File(...)):
    # Reset pointer in uploaded file to start
    jd_pdf.file.seek(0)
    jd_text = extract_text_from_pdf(jd_pdf.file)
    results = []

    # Connect to DB and fetch resume paths
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT resume_address FROM resume")  # Adjust column/table name
    resume_paths = cursor.fetchall()
    cursor.close()
    conn.close()

    for (resume_path,) in resume_paths:
        try:
            path_str = str(resume_path)  # Make sure it's a string
            with open(path_str, "rb") as f:
                resume_text = extract_text_from_pdf(f)
            score = compute_similarity(jd_text, resume_text)
            results.append({
                "resume_name": os.path.basename(path_str),
                "score": score
            })
        except Exception as e:
            results.append({
                "resume_name": os.path.basename(str(resume_path)),
                "score": None,
                "error": str(e)
            })


    return JSONResponse({"results": results})


# Here on I use a small LLM for skill extraction

def get_skill_extractor_llm():
    # Load a small LLM for skill extraction (adjust model as needed)
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=32)
    return HuggingFacePipeline(pipeline=pipe)

def extract_skills_from_query(query):
    llm = get_skill_extractor_llm()
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Extract a comma-separated list of key skills or technologies from the following job search query:\n"
            "{query}\nSkills:"
        ),
    )
    chain = prompt | llm
    result = chain.invoke({"query": query})
    # Parse skills from LLM output
    skills = [s.strip().lower() for s in result.split(",") if s.strip()]
    return set(skills)

@router.post("/search_resumes_cached")
async def search_resumes_cached(query: str):
    # Load FAISS index & metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)

    ids = metadata["ids"]
    names = metadata["names"]

    # Compute query embedding
    top_k = min(2, len(ids))  # Only top 2 resumes
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "resume_id": ids[idx],
            "resume_name": names[idx],
            # "score": float(100 - score)
        })

    return JSONResponse({"results": results})

class ResumeSearchRequest(BaseModel):
    query: str
    top_k: int = 5

@router.post("/search_existing_resumes")
async def search_existing_resumes(request: ResumeSearchRequest = Body(...)):
    """
    Search existing resumes in the database using the FAISS index.
    Returns the top_k most relevant resumes for the given query.
    """
    # Load FAISS index & metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)

    ids = metadata["ids"]
    names = metadata["names"]

    # Compute query embedding
    k = min(request.top_k, len(ids))
    query_emb = model.encode([request.query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=k)

    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "resume_id": ids[idx],
            "resume_name": names[idx],
            # "score": float(100 - score)
        })

    return JSONResponse({"results": results})
