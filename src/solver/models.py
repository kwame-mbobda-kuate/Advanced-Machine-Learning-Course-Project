from pymilvus import MilvusClient
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
import numpy as np

MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 16


class Reranker:

    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def rank(self, clues: List[str], answers: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            clues,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        targets = self.tokenizer(
            answers,
            return_tensors="pt",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=targets["input_ids"],
            )
        avg_nll = outputs.loss.item()
        num_target_tokens = targets["input_ids"].shape[1]
        sum_log_likelihood = -avg_nll * num_target_tokens
        return sum_log_likelihood


class Encoder:

    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(text)


class Retriever:

    def __init__(self, path: str):
        self.client = MilvusClient(path)

    def retrieve(
        self, vector: List[np.ndarray], k: int
    ) -> Tuple[List[List[str]], List[float]]:
        res = self.client.search(
            collection_name="answer",
            data=vector,
            limit=k,
            output_fields=["text"],
        )
        return [[item["text"] for item in sub_res] for sub_res in res], [
            [item["distance"] for item in sub_res] for sub_res in res
        ]
