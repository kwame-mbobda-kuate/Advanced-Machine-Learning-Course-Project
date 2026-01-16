import pandas as pd
import click
from solver.models import Encoder, Retriever

@click.command()
@click.argument("db_path")
@click.argument("model_name_or_path")
@click.argument("parquet_path")
@click.argument("batch_size", type=click.INT)
@click.option("--init", default=False, type=click.BOOL)
@click.option("--dim", default=0, type=click.INT)
def encode_answers(
    db_path: str,
    model_name_or_path: str,
    parquet_path: str,
    batch_size: int,
    init: bool = False,
    dim: int = 0,
) -> None:
    encoder = Encoder(model_name_or_path)
    retriever = Retriever(db_path)
    if init:
        retriever.init(dim)
    df = pd.read_parquet(parquet_path)
    encodings = encoder.encode(df["answer"], batch_size, show_progress_bar=True).tolist()
    retriever.insert(encodings, df["answer"])


if __name__ == '__main__':
    encode_answers()
