import torch
from datasets import load_dataset, DatasetDict
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
BASE_MODEL = "intfloat/multilingual-e5-small"
OUTPUT_PATH = "data/e5-two-tower"
HUB_MODEL_ID = "kwmk/e5-two-tower"

# Batch Sizes
TARGET_BATCH_SIZE = 64
MINI_BATCH_SIZE = 32
NUM_EPOCHS = 5

# E5 Prefixes
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Internal Router Keys (How the model sees the data)
# We will map 'clue' -> 'anchor' and 'answer' -> 'positive'
KEY_QUERY = "anchor"
KEY_DOC = "positive"

print(f"Initializing Two-Tower model from {BASE_MODEL}...")

# ---------------------------------------------------------
# 2. Initialize Divergent Two-Tower Model (Router)
# ---------------------------------------------------------
# Tower A: Query (Clue) Encoder
query_bert = models.Transformer(BASE_MODEL)
query_pool = models.Pooling(query_bert.get_word_embedding_dimension())
query_tower = SentenceTransformer(modules=[query_bert, query_pool])

# Tower B: Passage (Answer) Encoder
doc_bert = models.Transformer(BASE_MODEL)
doc_pool = models.Pooling(doc_bert.get_word_embedding_dimension())
doc_tower = SentenceTransformer(modules=[doc_bert, doc_pool])

# Router
router = models.Router(
    in_features_to_module={KEY_QUERY: query_tower, KEY_DOC: doc_tower}
)

model = SentenceTransformer(modules=[router])

# ---------------------------------------------------------
# 3. Load and Prepare Data
# ---------------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)

# --- Preprocessing for Training ---
print("Preprocessing training data...")


def preprocess_train(examples):
    # We assume columns are 'clue' and 'answer'
    # 1. Add E5 prefixes
    # 2. Rename to 'anchor' (KEY_QUERY) and 'positive' (KEY_DOC) for the Router
    return {
        KEY_QUERY: [QUERY_PREFIX + t for t in examples["clue"]],
        KEY_DOC: [PASSAGE_PREFIX + t for t in examples["answer"]],
    }


train_dataset = dataset["train"].map(preprocess_train, batched=True)
# Filter to keep only the columns the model expects
train_dataset = train_dataset.select_columns([KEY_QUERY, KEY_DOC])

# ---------------------------------------------------------
# 4. Prepare Evaluator (Handling Missing IDs)
# ---------------------------------------------------------
print("Preparing Evaluator from 'model_test'...")
val_data = dataset["model_test"]

queries = {}
corpus = {}
relevant_docs = {}

# Since 'model_test' only has text columns, we create IDs based on index.
# The 'corpus' will consist of all answers in the validation set.
for idx, row in enumerate(val_data):
    q_id = str(idx)
    doc_id = str(idx)  # Assuming 1-to-1 mapping in your test set

    # Store text with prefixes
    queries[q_id] = QUERY_PREFIX + row["clue"]
    corpus[doc_id] = PASSAGE_PREFIX + row["answer"]

    # Link them
    relevant_docs[q_id] = {doc_id}

evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    mrr_at_k=[10],
    name="model_test",
    main_score_function="mrr@10",
)

# ---------------------------------------------------------
# 5. Define Loss
# ---------------------------------------------------------
train_loss = losses.CachedMultipleNegativesRankingLoss(
    model=model, mini_batch_size=MINI_BATCH_SIZE, scale=100.0
)

# ---------------------------------------------------------
# 6. Training Configuration
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
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_model_test_mrr@10",
    greater_is_better=True,
    logging_steps=50,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # We manually push at the end
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
