import pandas as pd
import click
import tqdm
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
    for k in tqdm.tqdm(range(0, len(df), batch_size)):
        answers = df[k: k + batch_size]["answer"].tolist()
        encodings = encoder.encode(answers).tolist()
        retriever.insert(encodings, answers)


if __name__ == '__main__':
    encode_answers()