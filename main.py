# Run using gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import Optional
import os
import logging
logging.basicConfig(level=logging.INFO)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://vcjd5zdf-3000.asse.devtunnels.ms",
        "https://3eac-180-180-56-179.ngrok-free.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased', revision='main')
model = AutoModelForSequenceClassification.from_pretrained(
    'airesearch/wangchanberta-base-att-spm-uncased', 
    revision='finetuned@wisesight_sentiment', 
    num_labels=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://fluketj:9bKRqouZjHTn1qs4@sentimentfeedbackdb.t5iaj.mongodb.net/?retryWrites=true&w=majority&appName=SentimentFeedbackDB")
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Test the connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"MongoDB Connection Error: {e}")
    raise Exception("Failed to connect to MongoDB")

# Set up database and collection
db = client["sentiment_db"]
feedback_collection = db["feedback"]

# Request/Response models
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    score: float
    probabilities: dict

class FeedbackRequest(BaseModel):
    text: str
    predicted_sentiment: str
    corrected_sentiment: Optional[str] = None
    feedback: str  # "correct" or "incorrect"

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=416)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate softmax to get probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)[0].tolist()[:3]
        
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        if predicted_class == 3:
            predicted_class = 1
        
        # Map the predicted class to sentiment
        sentiment_mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
        sentiment = sentiment_mapping[predicted_class]
        
        # Prepare the response
        class_labels = ["Positive", "Neutral", "Negative"]
        probability_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
        
        # Get the score (probability of the predicted class)
        score = probabilities[predicted_class]
        
        return SentimentResponse(
            sentiment=sentiment, 
            score=score, 
            probabilities=probability_dict
        )
    
    except Exception as e:
        logging.error(f"Error predicting sentiment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Create feedback document with server-side timestamp
        feedback_data = feedback.dict()
        feedback_data["timestamp"] = datetime.utcnow()
        
        # Insert into MongoDB
        result = feedback_collection.insert_one(feedback_data)
        
        if result.inserted_id:
            return {"message": "Feedback submitted successfully", "id": str(result.inserted_id)}
        else:
            raise HTTPException(status_code=500, detail="Failed to insert feedback")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    model_status = "loaded" if model else "not loaded"
    try:
        client.admin.command('ping')
        return {"status": "healthy", "database": "connected", "model": model_status}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Database connection error")
