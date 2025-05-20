# app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

# Load model and tokenizer
model_path = './fine_tuned_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Sentiment labels
label_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Input schema
class InputText(BaseModel):
    text: str

# FastAPI app
app = FastAPI()

# Allow frontend to access the backend (CORS policy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predict endpoint
@app.post("/predict")
def predict_sentiment(input_data: InputText):
    inputs = tokenizer(input_data.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = round(probs[0][predicted_class].item(), 4)

    return {
        "label": label_map[predicted_class],
        "class_id": predicted_class,
        "confidence": confidence
    }
