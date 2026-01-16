from solver.models import Encoder, Retriever
import pandas as pd
import numpy as np
import click
import tqdm


@click.command()
@click.argument("db_path")
@click.argument("model_name_or_path")
@click.argument("parquet_path")
@click.argument("max_top_k", type=click.INT)
@click.argument("n_top_k", type=click.INT)
@click.argument("batch_size", type=click.INT)
def eval_accuracy(
    db_path: str,
    model_name_or_path: str,
    parquet_path: str,
    max_top_k: int,
    n_top_k: int,
    batch_size: int,
):
    encoder = Encoder(model_name_or_path)
    retriever = Retriever(db_path)
    df = pd.read_parquet(parquet_path)
    top_k = np.linspace(1, max_top_k, n_top_k).astype(int)
    ranks = []
    for j in tqdm.tqdm(range(0, len(parquet_path), batch_size)):
        split = df[j: j + batch_size].reset_index()
        encodings = encoder.encode(split["clue"])
        res, _ = retriever.retrieve(encodings, max_top_k)
        for sub_res, answer in zip(res, split["answer"]):
            ranks.append(sub_res.index(answer) if answer in sub_res else np.inf)
    acc = [0] * n_top_k
    for rank in ranks:
        for i in range(n_top_k):
            if rank < top_k[i]:
                acc[i] += 1
    acc = np.array(acc) / len(ranks)

    oracle_acc = 0
    all_known_answers = set()
    for known_answer in retriever.iter_all_data():
        all_known_answers.add(known_answer)
    for answer in df["answer"]:
        oracle_acc += answer in all_known_answers
    oracle_acc /= len(df)

    print(acc)
    print(oracle_acc)

if __name__ == "__main__":
    print(eval_accuracy())
