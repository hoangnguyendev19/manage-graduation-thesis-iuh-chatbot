from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import numpy as np

app = FastAPI()

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

def preprocess_text(text: str) -> str:
    return " ".join(word_tokenize(text))

def calculate_similarity(message: str, topics: list[Topic]) -> list[Topic]:
    documents = [preprocess_text(f"{topic.name} {topic.description} {topic.target} {topic.expected_result} {topic.standard_output} {topic.require_input} {topic.keywords}") for topic in topics]
    documents.append(preprocess_text(message))
    
    vectorizer = TfidfVectorizer().fit_transform(documents)
    user_vector = vectorizer[-1].toarray()
    topic_vectors = vectorizer[:-1].toarray()
    similarities = cosine_similarity(user_vector, topic_vectors)[0]
    
    sorted_indices = np.argsort(similarities)[::-1]
    return [topics[i] for i in sorted_indices if similarities[i] > 0.1]

@app.post("/api/v1/suggest-topics")
async def suggest_topics(request: SuggestionRequest):
    suggestions = calculate_similarity(request.message, request.topics)
    return suggestions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
