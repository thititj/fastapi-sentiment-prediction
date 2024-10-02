from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI()

# Load the tokenizer and model
# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(
                                 'airesearch/wangchanberta-base-att-spm-uncased',
                                 revision='main')

model = AutoModelForSequenceClassification.from_pretrained(
                                  'airesearch/wangchanberta-base-att-spm-uncased',
                                  revision='finetuned@wisesight_sentiment', num_labels=4)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    score: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted class and score
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        score = torch.softmax(outputs.logits, dim=1)[0, predicted_class].item()
        
        # Map the predicted class to sentiment
        sentiment_mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
        sentiment = sentiment_mapping[predicted_class]
        print(sentiment)
        
        return SentimentResponse(sentiment=sentiment, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

