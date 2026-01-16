import huggingface_hub
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
    losses,
    evaluation,
)
from sentence_transformers.training_args import BatchSamplers

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
DATASET_NAME = "kwmk/CrosswordClueAnswers"
BASE_MODEL = "dangvantuan/sentence-camembert-base"
OUTPUT_PATH = "data/scabert"
HUB_MODEL_ID = "kwmk/scabert"

huggingface_hub.login()

# Batch Sizes
TARGET_BATCH_SIZE = 2048
MINI_BATCH_SIZE = 1024
CORPUS_CHUNK_SIZE = 10_000
NUM_EPOCHS = 10
SCALE = 20

# E5 Prefixes
QUERY_PREFIX = ""
PASSAGE_PREFIX = ""

print(f"Initializing encoding model from {BASE_MODEL}...")

# ---------------------------------------------------------
# 2. Initialize Divergent Two-Tower Model
# ---------------------------------------------------------
model = SentenceTransformer(BASE_MODEL)

# ---------------------------------------------------------
# 3. Load and Prepare Data
# ---------------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)

# --- Preprocessing ---
# We do NOT rename columns here. We keep 'clue' and 'answer'.
# We only add the E5 prefixes to the text content.
print("Preprocessing training data (adding prefixes)...")


def add_prefixes(examples):
    return {
        "clue": [QUERY_PREFIX + t for t in examples["clue"]],
        "answer": [PASSAGE_PREFIX + t.lower() for t in examples["answer"]],
    }


train_dataset = dataset["train"].map(add_prefixes, batched=True)

# Ensure we keep the original columns needed for the mapping
train_dataset = train_dataset.select_columns(["clue", "answer"])

# ---------------------------------------------------------
# 4. Prepare Evaluator
# ---------------------------------------------------------
print("Preparing Evaluator...")
val_data = dataset["val"]

queries = {}
corpus = {}
relevant_docs = {}

# We manually prepare the evaluator dictionaries
for idx, row in enumerate(val_data):
    q_id = str(idx)
    doc_id = str(idx)

    # Note: InformationRetrievalEvaluator calls model.encode().
    # With a Router, it's best to ensure the inputs have prefixes.
    queries[q_id] = QUERY_PREFIX + row["clue"]
    corpus[doc_id] = PASSAGE_PREFIX + row["answer"].lower()
    relevant_docs[q_id] = {doc_id}

evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    mrr_at_k=[10, 100],
    corpus_chunk_size=CORPUS_CHUNK_SIZE,
    name="test",
)

# ---------------------------------------------------------
# 5. Define Loss
# ---------------------------------------------------------
train_loss = losses.CachedMultipleNegativesRankingLoss(
    model=model, mini_batch_size=MINI_BATCH_SIZE, scale=SCALE
)

# ---------------------------------------------------------
# 6. Training Configuration with Router Mapping
# ---------------------------------------------------------
args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TARGET_BATCH_SIZE,
    per_device_eval_batch_size=TARGET_BATCH_SIZE,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_test_cosine_mrr@100",
    greater_is_better=True,
    logging_steps=10,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

# ---------------------------------------------------------
# 7. Start Training
# ---------------------------------------------------------
print("Starting training...")
trainer.train()

# ---------------------------------------------------------
# 8. Save and Upload
# ---------------------------------------------------------
print(f"Saving locally to {OUTPUT_PATH}...")
model.save(OUTPUT_PATH)

print(f"Uploading model to Hugging Face Hub: {HUB_MODEL_ID}...")
model.push_to_hub(HUB_MODEL_ID, private=True)

print("Upload complete.")
