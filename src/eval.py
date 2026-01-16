from solver.models import Encoder, Retriever
import pandas as pd
import numpy as np
import click

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

    top_k_thresholds = np.linspace(1, max_top_k, n_top_k).astype(int)
    encodings = encoder.encode(df["clue"], batch_size, show_progress_bar=True)
    res, _ = retriever.retrieve(encodings, max_top_k)
    res_array = np.array(res)
    ground_truth = df["answer"].to_numpy()[:, None]
    matches_matrix = res_array == ground_truth
    acc = np.array([matches_matrix[:, :k].any(axis=1).mean() for k in top_k_thresholds])

    all_known_answers = set(retriever.iter_all_data())
    oracle_acc = df["answer"].isin(all_known_answers).mean()

    print(top_k_thresholds)
    print(acc)
    print(oracle_acc)

def eval_accuracy(
    db_path: str,
    retriever_model_name_or_path: str,
    reranker_model_name_or_path: str,
    grid_folder: str
):


if __name__ == "__main__":
    print(eval_accuracy())
