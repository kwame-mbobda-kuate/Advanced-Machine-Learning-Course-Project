from data import Grid
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


"""Pour corriger les grilles de mots croiser apres scrapping"""


def word_fits(layout, word, x, y, direction):
    """
    Vérifie si un mot rentre correctement dans la grille.
    """
    height = len(layout)
    width = len(layout[0])

    word = filter(str.isalpha, word)
    for i, letter in enumerate(word):
        letter = strip_accents(letter.upper())
        if direction == "H":
            xi, yi = x + i, y
        elif direction == "V":
            xi, yi = x, y + i
        else:
            raise ValueError("Direction invalide")

        # hors grille
        if xi < 0 or yi < 0 or yi >= height or xi >= width:
            return False

        # case noire
        if layout[yi][xi] == "@":
            return False

        # si la grille contient déjà une lettre différente
        if layout[yi][xi] != " " and layout[yi][xi] != letter:
            return False

    return True


def validate_grid(layout, clue_answer_pairs):
    """
    Vérifie si toute la grille est cohérente.
    """
    for entry in clue_answer_pairs:
        if not word_fits(
            layout, entry["ans"], entry["xy"][0], entry["xy"][1], entry["d"]
        ):
            return False
    return True


def test_xy_permutations(layout, clues):
    cases = {
        "aucune": lambda c: c,
        "swap_H": lambda c: {**c, "xy": c["xy"][::-1]} if c["d"] == "H" else c,
        "swap_V": lambda c: {**c, "xy": c["xy"][::-1]} if c["d"] == "V" else c,
        "swap_HV": lambda c: {**c, "xy": c["xy"][::-1]},
    }

    for name, transform in cases.items():
        transformed = [transform(c) for c in clues]

        if validate_grid(layout, transformed):
            return name, transformed

    return None, None


def fix_scraped_grid(data):
    mode, corrected_clues = test_xy_permutations(
        data["layout"], data["clue_answer_pairs"]
    )

    if not corrected_clues:
        raise ValueError("Aucune permutation valide trouvée")

    # data["meta"]["scrape_fix"] = mode
    data["clue_answer_pairs"] = corrected_clues

    return data


def overwrite_with_backup(path):
    shutil.copy(path, path + ".bak")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = fix_scraped_grid(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)