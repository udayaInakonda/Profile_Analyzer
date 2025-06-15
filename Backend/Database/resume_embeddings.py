# Run this separately, maybe as a background task or script

# Example extract_text_from_pdf function (using PyPDF2)
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
# ...existing code...
import logging

# Define DB_CONFIG
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Amzur@#123',
    'database': 'profile_analyzer'
}

def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_all_resume_texts(resume_paths):
    texts, ids, names = [], [], []
    for rid, path in resume_paths:
        try:
            with open(path, "rb") as f:
                text = extract_text_from_pdf(f)
                texts.append(text)
                ids.append(rid)
                names.append(os.path.basename(path))
        except:
            continue
    return texts, ids, names

if __name__ == "__main__":
    # Example: Fetch from MySQL
    import mysql.connector

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, resume_address FROM resume")
    resume_data = cursor.fetchall()
    cursor.close()
    conn.close()

    texts, ids, names = load_all_resume_texts(resume_data)
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings) # type: ignore
    faiss.write_index(index, "resume_index.faiss")

    # Save metadata
    with open("resume_metadata.pkl", "wb") as f:
        pickle.dump({"ids": ids, "names": names}, f)
