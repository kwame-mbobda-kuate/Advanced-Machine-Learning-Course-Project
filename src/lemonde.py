from data import Grid, GridFilling, Direction, ClueAnswerPair, Point
import requests
import os
import time
from datetime import datetime, timedelta
import json

with open("data/queries/le_monde_query.txt", "r") as f:
    query = f.read()


def convert_lemonde_crosssword_grid(crossword_grid: dict) -> Grid:
    game = crossword_grid["data"]["game"]
    author = game.get("author", None)
    grid = game["grid"]["grid"]
    grid_layout = [
        [GridFilling.BLOCKED] * grid["dimensions"][0]
        for _ in range(grid["dimensions"][1])
    ]
    clue_answer_pairs = []
    for definition in grid["definitions"]:
        clue = definition["text"].replace("  ", " ")
        answer = definition["solution"]
        start = Point(definition["coords"][1], definition["coords"][0])
        if definition["orientation"] == "horizontal":
            end = Point(definition["coords"][1], definition["coords"][0] + len(answer))
            direction = Direction.HORIZONTAL
            for j in range(start.x, end.x):
                grid_layout[start.y][j] = GridFilling.FREE
        else:
            end = Point(definition["coords"][1] + len(answer), definition["coords"][0])
            direction = Direction.VERTICAL
            for i in range(start.y, end.y):
                grid_layout[i][start.x] = GridFilling.FREE
        clue_answer_pair = ClueAnswerPair(clue, answer, start, end, direction)
        clue_answer_pairs.append(clue_answer_pair)
    return Grid(
        author=author, grid_layout=grid_layout, clue_answer_pairs=clue_answer_pairs
    )


def download_crossword_grid(date: datetime.date) -> Grid:
    url = "https://jeux-api.lemonde.fr/graphql"
    headers = {"content-type": "application/json"}
    variables = {"gameSlug": "mots-croises", "gridSlug": date.strftime("%d-%m-%Y")}
    payload = {"query": query, "variables": variables, "operationName": "GameDetail"}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        result = r.json()
        return convert_lemonde_crosssword_grid(result)
    else:
        return None


def download_all(delay: float, max_failures: int):
    date = datetime.today()
    os.makedirs("data/grids/le_monde", exist_ok=True)
    failures = 0
    while failures < max_failures:
        file_path = (f"data/grids/le_monde/{date.strftime('%d-%m-%Y')}.json",)
        if os.path.exists(file_path):
            date -= timedelta(days=1)
            failures = 0
            continue
        if grid := download_crossword_grid(date) is not None:
            with open(file_path, "w") as f:
                f.write(json.dumps(grid, cls=GridEncoder))
            failures = 0
        else:
            failures += 1
        date -= timedelta(days=1)
        time.sleep(delay)


if __name__ == "__main__":
    pass
