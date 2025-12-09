import asyncio
from data import Grid, GridFilling, Direction, ClueAnswerPair, Point
import aiohttp
import datetime
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


async def download_crossword_grid(date: datetime.date, session: aiohttp.ClientSession) -> Grid:
    url = "https://jeux-api.lemonde.fr/graphql"
    headers = {"content-type": "application/json"}
    variables = {"gameSlug": "mots-croises", "gridSlug": date.strftime("%d-%m-%Y")}
    payload = {
        "query": query,
        "variables": variables,
        "operationName": "GameDetail"
    }
    async with session.post(url, headers=headers, json=payload) as response:
        if response.status == 200:
            result = await response.json()
            return convert_lemonde_crosssword_grid(result)
        else:
            raise Exception(f"Request failed with status {response.status}")


if __name__ == "__main__":
    async def g():
        async with aiohttp.ClientSession() as session:
            return await download_crossword_grid(datetime.datetime.now(), session)
    print(asyncio.run(g()).clue_answer_pairs)
