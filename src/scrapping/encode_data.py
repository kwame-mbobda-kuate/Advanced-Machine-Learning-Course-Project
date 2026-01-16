import pandas as pd
from pymilvus import MilvusClient, DataType
from ..solver.models import Encoder


def encode_clues_answers(
    db_path: str,
    dimension: int,
    model_name_or_path: str,
    parquet_path: str,
    batch_size: int,
) -> None:
    client = MilvusClient(db_path)
    client.create_collection("answers", dimension)
    client.add_collection_field("answers", "text", DataType.VARCHAR, max_length=100)
    encoder = Encoder(model_name_or_path)
    df = pd.read_parquet(parquet_path)
    for k in range(0, len(df), batch_size):
        answers = df[k: k + batch_size]["answers"].tolist()
        encodings = encoder.encode(answers).tolist()
        data = []
        for j in range(batch_size):
            data.append({"id": j + k, "vector": encodings[j], "text": answers[j]})
        client.insert("answers", data)
