from data import Grid, GridFilling, Direction, ClueAnswerPair, Point
import json


def convert_lemonde_crosssword_grid(crossword_grid: str) -> Grid:
    game = json.loads(crossword_grid)["data"]["game"]
    author = game.get("author", None)
    grid = game["grid"]["grid"]
    grid_layout = [
        [GridFilling.BLOCKED] * grid["dimensions"][0]
        for _ in range(grid["dimensions"][1])
    ]
    clue_answer_pairs = []
    for definition in grid["definitions"]:
        clue = definition["text"]
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


if __name__ == "__main__":
    with open("data/test/le_monde_grid_1.json", "r") as f:
        print(convert_lemonde_crosssword_grid(f.read()).clue_answer_pairs)
    with open("data/test/le_monde_grid_mini_1.json", "r") as f:
        print(convert_lemonde_crosssword_grid(f.read()).clue_answer_pairs)
