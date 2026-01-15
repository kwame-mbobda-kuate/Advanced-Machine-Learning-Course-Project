import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
DATASET_NAME = "kwmk/CrosswordClueAnswers"
MODEL_ID = "moussaKam/barthez"
OUTPUT_DIR = "data/byt5"
HUB_MODEL_ID = "kwmk/byt5"

# Hyperparameters
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
NUM_EPOCHS = 10  # Generative models often converge faster than retrievers
LEARNING_RATE = 4e-5

print(f"Loading Model: {MODEL_ID}...")

# ---------------------------------------------------------
# 2. Load Tokenizer and Model
# ---------------------------------------------------------
# ByT5 uses a specific tokenizer that processes raw bytes
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the T5 model for conditional generation (Seq2Seq)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# ---------------------------------------------------------
# 3. Load and Preprocess Data
# ---------------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)


# Preprocessing function
def preprocess_function(examples):
    inputs = examples["clue"]
    targets = [str(t).lower() for t in examples["answer"]]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True
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


def equal_chars(s1, s2):
    m = min(len(s1), len(s2))
    M = max(len(s2), len(s2))
    if M == 0:
        return 0
    return sum(s1[i] == s2[i] for i in range(m)) / M


def normalize(s):
    return "".join(filter(str.isalpha, s.lower()))


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
    decoded_preds = [normalize(pred) for pred in decoded_preds]
    decoded_labels = [normalize(label) for label in decoded_labels]

    # Calculate Exact Match Accuracy
    matches = [equal_chars(p, l) for p, l in zip(decoded_preds, decoded_labels)]
    accuracy = sum(matches) / len(matches)

    return {"accuracy": accuracy}


# ---------------------------------------------------------
# 5. Training Configuration
# ---------------------------------------------------------
# Data Collator handles padding dynamically to the longest sequence in the batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    generation_max_length=MAX_TARGET_LENGTH,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
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
