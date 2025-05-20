# from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
# from transformers import TrainerCallback
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# import pandas as pd
# import torch
# import torch.nn as nn

# # Load the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)

# # Load and prepare the dataset
# df = pd.read_excel('full_multilingual_dataset.xlsx')  # this includes your 250 original + 250 generated = 500
# texts = df['text'].tolist()
# labels = df['label'].tolist()

# # Tokenize
# def tokenize(batch):
#     return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

# data = Dataset.from_pandas(pd.DataFrame({'text': texts, 'label': labels}))
# data = data.train_test_split(test_size=0.2, seed=42)
# data = data.map(tokenize, batched=True)

# # Custom Trainer with class weights
# class WeightedTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         weights = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.3]).to(model.device)
#         loss_fct = nn.CrossEntropyLoss(weight=weights)
#         loss = loss_fct(logits, labels)

#         return (loss, outputs) if return_outputs else loss

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./fine_tuned_model",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss"
# )

# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=data['train'],
#     eval_dataset=data['test'],
#     tokenizer=tokenizer,
# )

# trainer.train()

# model.save_pretrained("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_model")



import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load dataset from Excel file (with labels from 0 to 4)
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df[['text', 'label']].dropna()

    # Ensure labels are strictly between 0 and 4 (no conversion needed)
    df = df[df['label'].between(0, 4)]
    
    # Debugging print statement
    print("Unique labels in dataset before training:", df['label'].unique())

    return df

# Preprocess data
def preprocess_data(df, tokenizer, max_length=128):
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

# Load and split dataset into train, validation, and test sets
def prepare_dataset(file_path, tokenizer):
    df = load_dataset(file_path)

    # Split into 80% train and 20% test
    train_texts, test_texts = train_test_split(df, test_size=0.2, random_state=42)

    # Further split train_texts into 90% train and 10% validation
    train_texts, val_texts = train_test_split(train_texts, test_size=0.1, random_state=42)

    # Save the test data to an Excel file (no need to adjust labels)
    test_texts.to_excel('testing_data_multilingual.xlsx', index=False)

    # Preprocess datasets
    train_dataset = preprocess_data(train_texts, tokenizer)
    val_dataset = preprocess_data(val_texts, tokenizer)
    test_dataset = preprocess_data(test_texts, tokenizer)

    # Save the validation data to an Excel file
    val_texts.to_excel('validation_data_multilingual.xlsx', index=False)

    return train_dataset, val_dataset, test_dataset

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Prepare datasets
train_dataset, val_dataset, test_dataset = prepare_dataset('multilingual_dataset.xlsx', tokenizer)

# Define the model (using 5 labels from 0 to 4)
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', num_labels=5)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Evaluate the model on the test dataset
results = trainer.evaluate(test_dataset)
print(results)
