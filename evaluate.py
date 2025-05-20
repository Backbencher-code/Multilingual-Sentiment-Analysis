import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')
model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
model.eval()

# Load test data
def load_test_data(file_path):
    df = pd.read_excel(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

# Create DataLoader
def create_dataloader(texts, labels, tokenizer, batch_size=16, max_length=128):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size)

# Predict
def predict(model, dataloader):
    preds, true = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true.extend(labels.cpu().numpy())
    return preds, true

# Run evaluation
if __name__ == "__main__":
    test_texts, test_labels = load_test_data('testing_data_multilingual.xlsx')
    test_loader = create_dataloader(test_texts, test_labels, tokenizer)
    y_pred, y_true = predict(model, test_loader)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=2))
