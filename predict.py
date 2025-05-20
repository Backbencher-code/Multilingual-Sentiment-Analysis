import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')
model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
model.eval()

# Sentiment label map
label_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Function to predict sentiment of multilingual input
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    return predicted_class, label_map[predicted_class], confidence

# Example usage
if __name__ == "__main__":
    print("Enter a multilingual sentence (mix of English, Hindi, French):")
    text = input("> ")
    class_id, label_text, confidence = predict(text)
    print(f"\nPredicted Sentiment: {label_text} (Label {class_id})")
    print(f"Confidence Score: {confidence:.2f}")
