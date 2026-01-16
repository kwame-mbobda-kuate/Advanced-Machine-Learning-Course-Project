import json
import shutil
from pathlib import Path
import time
import unicodedata


def normalize_clue(clue: str) -> str:
    clue = clue.strip()
    clue = (
        clue.strip()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("…", "...")
        .replace("–", "-")
        .replace("—", "-")
        .replace("- ", "-")
    )
    clue = clue.rstrip(".").strip()
    if "... " in clue:
        clue = clue.replace("... ", " ").strip()
    if clue == clue.upper():
        clue = clue.capitalize()
    return clue


def normalize_answer(ans: str) -> str:
    return "".join(
        filter(str.isalpha, unicodedata.normalize("NFD", ans.lower()))
    ).replace("œ", "oe")


def grid_to_crossword(grid_obj):
    height = len(grid_obj.layout)
    width = len(grid_obj.layout[0])

    letter_grid = [["" for _ in range(width)] for _ in range(height)]
    number_grid = [["" for _ in range(width)] for _ in range(height)]

    across = {}
    down = {}
    cell_numbers = {}
    current_number = 1

    # 1️⃣ Numérotation + placement des lettres
    for clue in grid_obj.clue_answer_pairs:
        y, x = clue.start.y, clue.start.x
        d = clue.direction.to_code()

        if (y, x) not in cell_numbers:
            cell_numbers[(y, x)] = str(current_number)
            current_number += 1

        num = cell_numbers[(y, x)]
        number_grid[y][x] = num

        for i, letter in enumerate(clue.answer.upper()):
            if d == "H":
                letter_grid[y][x + i] = letter
            else:
                letter_grid[y + i][x] = letter

        if d == "H":
            across[num] = [clue.clue, clue.answer]
        else:
            down[num] = [clue.clue, clue.answer]

    # 2️⃣ Construction de la grille finale
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            cell = grid_obj.layout[y][x]

            # CASE NOIRE
            if cell in ("@", ","):
                row.append("BLACK")
            else:
                row.append([number_grid[y][x], letter_grid[y][x]])
        grid.append(row)

    # 3️⃣ Format final EXACT attendu
    return {
        "metadata": {"date": None, "rows": height, "cols": width},
        "clues": {"across": across, "down": down},
        "grid": grid,
    }
