from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
from keybert import KeyBERT
from transformers import pipeline
import logging
import re

router = APIRouter()

_sentence_model = None
_kw_model = None
_flan_tokenizer = None
_flan_model = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(get_sentence_model())
    return _kw_model

def get_flan_model():
    global _flan_tokenizer, _flan_model
    if _flan_tokenizer is None or _flan_model is None:
        model_name = "google/flan-t5-small"
        _flan_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _flan_tokenizer, _flan_model

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def compute_similarity(text1, text2):
    model = get_sentence_model()
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    score = max(0, min(100, int(sim * 100)))
    return score

def extract_skills_with_keybert(text: str, top_n=15) -> list:
    kw_model = get_kw_model()
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n
    )
    # Remove duplicates and keep only the keyword string
    seen = set()
    skills = []
    for kw, _ in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            skills.append(kw_lower)
            seen.add(kw_lower)
    return skills

def normalize_skills(skills):
    # Split phrases into words, remove stopwords, and deduplicate
    stopwords = {"in", "of", "and", "with", "for", "to", "the", "a", "an"}
    normalized = set()
    for skill in skills:
        words = re.split(r'\W+', skill.lower())
        for word in words:
            if word and word not in stopwords:
                normalized.add(word)
    return normalized

def generate_reason_with_flan(jd_skills, resume_skills, matched_skills, missing_skills, score):
    prompt = (
        f"You are analyzing how well a candidate’s skills match a job description.\n\n"
        f"Job Description Skills: {', '.join(jd_skills[:8])}\n"
        f"Resume Skills: {', '.join(resume_skills[:8])}\n"
        f"Matched Skills: {', '.join(matched_skills[:8])}\n"
        f"Missing Skills: {', '.join(missing_skills[:8])}\n"
        f"Skill Match Score: {score} out of 100\n\n"
        "Write a short, helpful summary for the candidate. "
        "Explain why the score is high or low, highlight strengths, and give specific suggestions for improvement."
    )
    tokenizer, model = get_flan_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            use_cache=False
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()

@router.post("/analyze")
async def analyze(jd_pdf: UploadFile = File(...), resume_pdf: UploadFile = File(...)):
    jd_text = extract_text_from_pdf(jd_pdf.file)
    resume_text = extract_text_from_pdf(resume_pdf.file)
    score = compute_similarity(jd_text, resume_text)
    return JSONResponse({"score": score})

@router.post("/insights")
async def insights(jd_pdf: UploadFile = File(...), resume_pdf: UploadFile = File(...)):
    jd_text = extract_text_from_pdf(jd_pdf.file)
    resume_text = extract_text_from_pdf(resume_pdf.file)
    score = compute_similarity(jd_text, resume_text)

    jd_skills_raw = extract_skills_with_keybert(jd_text)
    resume_skills_raw = extract_skills_with_keybert(resume_text)
    jd_skills = normalize_skills(jd_skills_raw)
    resume_skills = normalize_skills(resume_skills_raw)
    matched_skills = sorted(list(jd_skills & resume_skills))
    missing_skills = sorted(list(jd_skills - resume_skills))

    reason = generate_reason_with_flan(
        jd_skills_raw, resume_skills_raw, matched_skills, missing_skills, score
    )

    return JSONResponse({
        "score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "reason": reason
    })