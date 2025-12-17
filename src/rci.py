from data import *
from py_mini_racer import MiniRacer
from typing import List, Tuple
import json
import requests
import os
import time


def parse_RCI_crossword_grid(crossword_grid: str) -> dict:
    ctx = MiniRacer()
    ctx.eval(crossword_grid)
    return json.loads(ctx.eval("JSON.stringify(gamedata)"))


def transpose(grid: List[List[str]]) -> List[List[str]]:
    width, height = len(grid[0][0]), len(grid)
    transposed = []
    for j in range(width):
        v_line = "".join(grid[i][0][j] for i in range(height))
        transposed.append([v_line])
    return transposed


def extract_clue_answer_pairs_RCI_crossword_grid(
    grid: List[List[str]], clues: List[List[str]], direction: Direction
) -> List[ClueAnswerPair]:
    clue_answer_pairs = []
    for i, line in enumerate(grid):
        potential_answers = line[0].split("x")
        x = 0
        clue_number = 0
        for j, potential_answer in enumerate(potential_answers):
            if len(potential_answer) > 1:
                clue_answer_pair = ClueAnswerPair(
                    clues[i][clue_number],
                    potential_answer,
                    Point(x, i),
                    Point(x + len(potential_answer), i),
                    direction,
                )
                clue_number += 1
                clue_answer_pairs.append(clue_answer_pair)
            x += len(potential_answer) + 1
    return clue_answer_pairs


def convert_RCI_crossword_grid(crossword_grid: str) -> Grid:
    json_crossword_grid = parse_RCI_crossword_grid(crossword_grid)
    clues_h = extract_clue_answer_pairs_RCI_crossword_grid(
        json_crossword_grid["grille"],
        json_crossword_grid["definitionsh"],
        Direction.HORIZONTAL,
    )
    clues_v = extract_clue_answer_pairs_RCI_crossword_grid(
        transpose(json_crossword_grid["grille"]),
        json_crossword_grid["definitionsv"],
        Direction.VERTICAL,
    )
    clue_answer_pairs = clues_h + clues_v
    grid_layout = []
    for line in json_crossword_grid["grille"]:
        layout_line = [
            GridFilling.FREE if char.isupper() else GridFilling.BLOCKED
            for char in line[0]
        ]
        grid_layout.append(layout_line)
    difficulty = json_crossword_grid.get("force", None)
    if difficulty:
        difficulty = int(difficulty)
    return Grid(
        grid_layout=grid_layout,
        clue_answer_pairs=clue_answer_pairs,
        difficulty=difficulty,
    )


def next_lower_char(l: str, i: int):
    j = i
    while j < len(l) and not l[j].islower():
        j += 1
    return j


def extract_hor(grid: List[List[str]], clue: str, i: int, j: int) -> ClueAnswerPair:
    line = grid[i]
    k = next_lower_char(line, j)
    answer = line[j:k]
    return ClueAnswerPair(clue, answer, Point(i, j), Point(i, k), Direction.HORIZONTAL)


def extract_ver(grid: List[List[str]], clue: str, i: int, j: int) -> ClueAnswerPair:
    line = "".join(grid[k][j] for k in range(len(grid)))
    k = next_lower_char(line, i)
    answer = line[i:k]
    return ClueAnswerPair(clue, answer, Point(i, j), Point(k, j), Direction.VERTICAL)


def join_clue(l: List[str]) -> str:
    joined_clue = ""
    for split_clue in l:
        if split_clue.endswith("–"):
            # Dashes are used for hyphenation
            joined_clue += split_clue[:-1]
        else:
            joined_clue += split_clue + " "
    return joined_clue.strip()


def convert_RCI_arrow_crossword_grid(crossword_grid: str):
    json_crossword_grid = parse_RCI_crossword_grid(crossword_grid)
    difficulty = None
    if json_crossword_grid["force"]:
        difficulty = int(json_crossword_grid["force"])
    grille = json_crossword_grid["grille"]
    clues = json_crossword_grid["definitions"]
    width = len(grille[0])
    height = len(grille)
    clue_answer_pairs = []
    clue_number = 0
    hor_right = "a"
    hor_right_ver_down = "hiefg"
    hor_right_hor_down = "qrosp"
    ver_right_hor_down = "txvwu"
    ver_right_ver_down = "nmlkj"
    ver_right = "c"
    ver_down = "b"
    hor_down = "d"
    filled = "z"
    # y?
    # z : voir https://www.rcijeux.fr/drupal_game/lci/mfleches/grids/5439.mfj
    for i, line in enumerate(grille):
        j = 0
        while j < width:
            j = next_lower_char(line, j)
            if j >= width:
                break
            char = line[j]
            if char in filled:
                continue
            if (
                char
                not in hor_right
                + hor_right_ver_down
                + hor_right_hor_down
                + ver_right_hor_down
                + ver_right_ver_down
                + ver_right
                + ver_down
                + hor_down
            ):
                raise GridConversionError(f"Unknown clue position indicator: {char}.")
            if char in hor_right + hor_right_hor_down + hor_right_ver_down:
                # Retrieval of the horizontal clue on the right
                clue_answer_pairs.append(
                    extract_hor(
                        grille,
                        join_clue(clues[clue_number]),
                        i,
                        j + 1,
                    )
                )
                clue_number += 1
            if char in ver_right + ver_right_hor_down + ver_right_ver_down:
                # Retrieval of the vertical clue on the right
                clue_answer_pairs.append(
                    extract_ver(
                        grille,
                        join_clue(clues[clue_number]),
                        i,
                        j + 1,
                    )
                )
                clue_number += 1
            if char in ver_down + hor_right_ver_down + ver_right_ver_down:
                # Retrieval of the vertical clue on the bottom
                clue_answer_pairs.append(
                    extract_ver(
                        grille,
                        join_clue(clues[clue_number]),
                        i + 1,
                        j,
                    )
                )
                clue_number += 1
            if char in hor_down + ver_right_hor_down + hor_right_hor_down:
                # Retrieval of the horizontal clue on the bottom
                clue_answer_pairs.append(
                    extract_hor(
                        grille,
                        join_clue(clues[clue_number]),
                        i + 1,
                        j,
                    )
                )
                clue_number += 1
            j += 1
    grid_layout = []
    for line in grille:
        layout_line = [
            GridFilling.FREE if char.isupper() else GridFilling.BLOCKED
            for char in line[0]
        ]
        grid_layout.append(layout_line)
    difficulty = json_crossword_grid.get("force", None)
    if difficulty is not None:
        difficulty = int(difficulty)
    return Grid(
        grid_layout=grid_layout,
        clue_answer_pairs=clue_answer_pairs,
        difficulty=difficulty,
    )


endpoints = [
    ("https://www.rcijeux.fr/drupal_game/lacroix/mcroises1/grids/gary{}.mcj", 8779),
    ("https://www.rcijeux.fr/drupal_game/lacroix/mcroises3/grids/olivier{}.mcj", 311),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mcroises/grids/mcroises_4_{}.mcj",
        1423,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mcroises/grids/mcroises_3_{}.mcj",
        1423,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mcroises/grids/mcroises_2_{}.mcj",
        1423,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mcroises/grids/mcroises_1_{}.mcj",
        1423,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mfleches/grids/mfleches_4_{}.mfj",
        3789,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mfleches/grids/mfleches_3_{}.mfj",
        3789,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mfleches/grids/mfleches_2_{}.mfj",
        3789,
    ),
    (
        "https://www.rcijeux.fr/drupal_game/notretemps/mfleches/grids/mfleches_1_{}.mfj",
        3789,
    ),
    ("https://www.rcijeux.fr/drupal_game/lebelage/mfleches/grids/{}.mfj", 1442),
    ("https://www.rcijeux.fr/drupal_game/lebelage/mcroises/grids/{}.mcj", 1442),
    ("https://www.rcijeux.fr/drupal_game/nrco/mfleches/grids/{}.mfj", 2002),
    ("https://www.rcijeux.fr/drupal_game/nrco/mcroises/grids/{}.mcj", 2002),
    ("https://www.rcijeux.fr/drupal_game/lci/mcroises/grids/{}.mcj", 4783),
    ("https://www.rcijeux.fr/drupal_game/lci/mfleches/grids/{}.mfj", 5439),
    ("https://www.rcijeux.fr/drupal_game/cnews/mfleches/grids/{}.mfj", 1939),
    ("https://www.rcijeux.fr/drupal_game/cnews/mfleches/grids/{}.mcj", 1939),
    ("https://www.rcijeux.fr/drupal_game/maxi/mcroises/grids/{}.mcj", 4550),
    ("https://www.rcijeux.fr/drupal_game/maxi/mcroises/grids/{}.mfj", 4550),
]


def download_grid(endpoint: str):
    r = requests.get(endpoint)
    if r.status_code == 200:
        text = r.text
        if endpoint.endswith("mfj"):
            grid = convert_RCI_arrow_crossword_grid(text)
        if endpoint.endswith("mcj"):
            grid = convert_RCI_crossword_grid(text)
        return grid
    else:
        return None


def download_all_endpoints(endpoints: List[Tuple[str, int]], delay: float, max_failures: int):
    for endpoint, end_index in endpoints:
        provider = endpoint.split("/")[-4]
        os.makedirs(f"data/grids/{provider}", exist_ok=True)            
        k = end_index
        for offset in (1, -1):
            failures = 0
            while failures < max_failures:
                file_path = f"data/grids/{provider}/{k}.json"
                if os.path.exists(file_path):
                    k += offset
                    failures = 0
                    continue
                if grid := download_grid(endpoint.format(k)) is not None:
                    grids.append((grid, k))
                    failures = 0
                    with open(file_path, "w") as f:
                        f.write(json.dumps(grid, cls=GridEncoder))
                else:
                    failures += 1
                k += offset
                time.sleep(delay)
            k = end_index - 1

if __name__ == "__main__":
    pass