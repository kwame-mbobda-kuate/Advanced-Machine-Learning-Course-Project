from pymilvus import MilvusClient, DataType
from typing import List, Union, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
import numpy as np

MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 16
DB_BATCH_SIZE = 10_000
BATCH_SIZE = 4096


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

    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = BATCH_SIZE,
        show_progress_bar=False,
    ) -> np.ndarray:
        return self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )


class Retriever:
    def __init__(self, path: str):
        self.client = MilvusClient(path)
        self.collection_name = "answers"

    def init(self, dim: int):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=100)

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(self.collection_name, schema=schema)

    def insert(self, encodings, answers, batch_size=BATCH_SIZE):
        def _send_batch(batch_df):
            self.client.insert(self.collection_name, data=batch_df.to_dict("records"))

        df = pd.DataFrame(
            {
                "vector": encodings,
                "text": answers,
            }
        )

        total_rows = len(df)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i : i + batch_size]
                futures.append(executor.submit(_send_batch, batch_df))
            for f in futures:
                f.result()
        self.build_index()

    def build_index(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )

        self.client.create_index(self.collection_name, index_params=index_params)
        self.client.load_collection(self.collection_name)

    def retrieve(
        self, vector: List[np.ndarray], k: int
    ) -> Tuple[List[List[str]], List[float]]:

        res = self.client.search(
            collection_name=self.collection_name,
            data=vector,
            limit=k,
            output_fields=["text"],
        )

        return [[item["text"] for item in sub_res] for sub_res in res], [
            [item["distance"] for item in sub_res] for sub_res in res
        ]
