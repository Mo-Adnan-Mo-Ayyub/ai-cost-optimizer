# train.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load annotated data
df = pd.read_csv("annotated_data.csv")  # Ensure your CSV has the expected columns
dataset = Dataset.from_pandas(df)

# Initialize tokenizer and model (example using T5)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = examples["code"]
    # Concatenate the annotated outputs into a target string.
    targets = (
        '{"cloud_cost_inefficiencies": ' + examples["cloud_cost_inefficiencies"] +
        ', "security_vulnerabilities": ' + examples["security_vulnerabilities"] +
        ', "optimization_suggestions": ' + examples["optimization_suggestions"] + "}"
    )
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=256)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    predict_with_generate=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # In practice, split into train/test
    tokenizer=tokenizer,
)

# Start training
trainer.train()
