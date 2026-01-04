import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
DATASET_NAME = "kwmk/CrosswordClueAnswers"
MODEL_ID = "google/byt5-small"
OUTPUT_DIR = "data/byt5"
HUB_MODEL_ID = "kwmk/byt5"

# Hyperparameters
BATCH_SIZE = 16  # ByT5 sequences are longer (bytes), so we keep batch size moderate
NUM_EPOCHS = 5  # Generative models often converge faster than retrievers
LEARNING_RATE = 8e-4  # T5 models often like slightly higher LR (e.g. 3e-4 to 1e-3)

# Max lengths (in Bytes/Characters)
# ByT5 uses roughly 1 token per character.
MAX_INPUT_LENGTH = 512  # Max length for the Clue
MAX_TARGET_LENGTH = 128  # Max length for the Answer

print(f"Loading Model: {MODEL_ID}...")

# ---------------------------------------------------------
# 2. Load Tokenizer and Model
# ---------------------------------------------------------
# ByT5 uses a specific tokenizer that processes raw bytes
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the T5 model for conditional generation (Seq2Seq)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

# ---------------------------------------------------------
# 3. Load and Preprocess Data
# ---------------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)


# Preprocessing function
def preprocess_function(examples):
    # T5 models usually work best with a task prefix, though ByT5 is flexible.
    # We format inputs as "clue: <clue_text>"
    inputs = [f"clue: {clue}" for clue in examples["clue"]]
    targets = [str(answer) for answer in examples["answer"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,  # We pad dynamically in the collator
    )

    # Tokenize targets (labels)
    # modern transformers use `text_target` argument
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,  # Remove raw text columns to save RAM
)

# ---------------------------------------------------------
# 4. Metrics for Evaluation
# ---------------------------------------------------------
# We define a simple "Exact Match" accuracy


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode generated predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Calculate Exact Match Accuracy
    matches = [1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)]
    accuracy = sum(matches) / len(matches)

    return {"accuracy": accuracy}


# ---------------------------------------------------------
# 5. Training Configuration
# ---------------------------------------------------------
# Data Collator handles padding dynamically to the longest sequence in the batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,  # Crucial for calculating metrics during eval
    fp16=True,  # Faster training on GPU
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    # Generation config for evaluation
    generation_max_length=MAX_TARGET_LENGTH,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["model_test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# 6. Train
# ---------------------------------------------------------
print("Starting training...")
trainer.train()

# ---------------------------------------------------------
# 7. Save and Upload
# ---------------------------------------------------------
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)

print(f"Uploading to Hugging Face Hub: {HUB_MODEL_ID}...")
# This pushes the model, tokenizer, and training args
trainer.push_to_hub(repo_id=HUB_MODEL_ID, private=True)

print("Done.")
