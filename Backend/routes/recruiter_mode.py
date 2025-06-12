from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
import mysql.connector
import os
import faiss
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

router = APIRouter()
model = SentenceTransformer('all-MiniLM-L6-v2')

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Amzur@#123',
    'database': 'profile_analyzer'
}

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

@router.post("/search_resumes")
async def search_resumes(query: str):
    # Extract relevant skills from the query using LLM
    query_skills = extract_skills_from_query(query)

    # Connect to DB and fetch resume paths
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT resume_address, id FROM resume")
    resume_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    resume_texts = []
    resume_names = []
    resume_ids = []
    for path, rid in resume_rows:
        try:
            with open(str(path), "rb") as f:
                text = extract_text_from_pdf(f)
            # Only consider resumes that mention at least one query skill
            if any(skill in text.lower() for skill in query_skills):
                resume_texts.append(text)
                resume_names.append(os.path.basename(str(path)))
                resume_ids.append(rid)
        except Exception:
            continue

    # If no query_skills were extracted, fallback: consider all resumes
    if not query_skills:
        for path, rid in resume_rows:
            try:
                with open(str(path), "rb") as f:
                    text = extract_text_from_pdf(f)
                resume_texts.append(text)
                resume_names.append(os.path.basename(str(path)))
                resume_ids.append(rid)
            except Exception:
                continue

    # If still no resumes, return at least the query skills
    if not resume_texts:
        return JSONResponse({"results": [], "querySkills": list(query_skills)})

    # Compute embeddings for resumes
    embeddings = model.encode(resume_texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Compute embedding for query
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=min(10, len(resume_texts)))  # Top 10 matches

    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "resume_id": resume_ids[idx],
            "resume_name": resume_names[idx],
            "score": float(100 - score)  # Lower L2 means more similar; invert for score
        })

    return JSONResponse({"results": results, "querySkills": list(query_skills)})
