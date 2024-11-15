from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import fitz  # PyMuPDF for faster PDF processing
import os
import numpy as np
import tempfile
import requests

app = FastAPI()

# Class definitions for topics
class Topic(BaseModel):
    id: str
    name: str
    description: str
    target: str
    expected_result: str
    standard_output: str
    require_input: str
    keywords: str

class SuggestionRequest(BaseModel):
    message: str
    topics: list[Topic]

# Efficient text extraction from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text") or ""
    return text

# Calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return similarity

# Reference folder path
REFERENCE_FOLDER = "D:/Document_University/KLTN/Docs_CoHanh/KLTN_Documents"

# Yield file names and paths from reference folder
def get_reference_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".pdf"):
            yield filename, file_path

# Topic suggestion endpoint
@app.post("/api/v1/suggest-topics")
async def suggest_topics(request: SuggestionRequest):
    message = request.message
    topics = request.topics
    
    # Process topics for TF-IDF
    documents = [
        " ".join(word_tokenize(f"{topic.name} {topic.description} {topic.target} {topic.expected_result} {topic.standard_output} {topic.require_input} {topic.keywords}"))
        for topic in topics
    ]
    documents.append(" ".join(word_tokenize(message)))
    
    vectorizer = TfidfVectorizer().fit_transform(documents)
    user_vector = vectorizer[-1].toarray()
    topic_vectors = vectorizer[:-1].toarray()
    similarities = cosine_similarity(user_vector, topic_vectors)[0]
    
    sorted_indices = np.argsort(similarities)[::-1]
    suggestions = [topics[i] for i in sorted_indices if similarities[i] > 0.1]
    
    return suggestions

# Plagiarism check endpoint
@app.post("/api/v1/check-plagiarism")
async def check_plagiarism(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        uploaded_text = extract_text_from_pdf(temp_file_path)
        os.remove(temp_file_path)

        results = []
        for filename, ref_file_path in get_reference_files(REFERENCE_FOLDER):
            ref_text = extract_text_from_pdf(ref_file_path)
            similarity = calculate_similarity(uploaded_text, ref_text)
            results.append({"reference": filename, "similarity": similarity})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return {"plagiarism_results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Detection API endpoint and key
AI_DETECTION_API_URL = "https://api.sapling.ai/api/v1/aidetect" # sapling.ai
API_KEY = "7RLION9EJPPSH8OAB5HCR4RS3T4GKMOJ"

# Send text to AI detection API
def check_ai_content(text):
    response = requests.post(
        AI_DETECTION_API_URL,
        json={"key": API_KEY, "text": text}
    )
    print(response.json())
    if response.status_code == 200:
        return response.json()
    else:
        
        raise HTTPException(status_code=500, detail="Failed to connect to AI detection API")

# Endpoint to check AI-generated content in PDF
@app.post("/api/v1/check-ai-content")
async def check_ai_content_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        extracted_text = extract_text_from_pdf(temp_file_path)
        os.remove(temp_file_path)

        result = check_ai_content(extracted_text)
        return {"ai_detection_result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
