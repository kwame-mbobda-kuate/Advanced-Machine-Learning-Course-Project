import os
import requests
import xml.etree.ElementTree as ET
import datetime
import concurrent.futures
from typing import Iterator
from tqdm import tqdm  # <--- NEW IMPORT

# Assumes these are in your 'data.py'
from data import (
    Grid,
    GridFilling,
    Direction,
    ClueAnswerPair,
    Point,
    GridConversionError,
)

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "./lefigaro_grids"
BASE_URL = "https://web.keesing.com/content/getxml?clientid=lefigaro&puzzleid=KFR-"
SEED_ID = 11514557
MAX_WORKERS = 100  # Number of simultaneous threads
SEARCH_RADIUS = 514557  # How far to check in each direction


# ==========================================
# 1. CONVERSION LOGIC
# ==========================================
def convert_lefigaro_arrow_crossword_grid(crossword_grid: str) -> Grid:
    if "Invalid xml request" in crossword_grid:
        raise GridConversionError("Incorrect KFR number.")

    try:
        root = ET.fromstring(crossword_grid)
    except ET.ParseError:
        raise GridConversionError("Response is not valid XML.")

    if root.attrib.get("type") != "Arrowword":
        raise GridConversionError("Not an Arrowword grid.")

    try:
        width = int(root.attrib["width"])
        height = int(root.attrib["height"])
        difficulty = int(root.attrib["difficulty"])
        date_str = root.attrib.get("exported")
        date = datetime.datetime.fromisoformat(date_str) if date_str else None
    except (ValueError, KeyError) as e:
        raise GridConversionError(f"Metadata error: {e}")

    clues = {Direction.HORIZONTAL: {}, Direction.VERTICAL: {}}
    grid_layout = [[GridFilling.BLOCKED] * width for _ in range(height)]
    clue_answer_pairs = []

    grid_node = root.find("grid")
    if grid_node is None:
        raise GridConversionError("No <grid> tag found.")

    for cell in grid_node.iter("cell"):
        x, y = int(cell.attrib["x"]), int(cell.attrib["y"])
        if cell.attrib.get("fillable") == "1":
            grid_layout[y][x] = GridFilling.FREE
        for clue in cell.iter("clue"):
            clue_text = (clue.text if clue.text else "").replace("\\", "")
            arrow = clue.attrib.get("arrow")
            if arrow in [
                "arrowdownrightbottom",
                "arrowrighttop",
                "arrowdownright",
                "arrowright",
            ]:
                direction = Direction.HORIZONTAL
            elif arrow in ["arrowdownbottom", "arrowrightdowntop", "arrowdown"]:
                direction = Direction.VERTICAL
            else:
                raise GridConversionError(f"Unknown arrow: {arrow}")
            clues[direction][clue.attrib["wordindex"]] = clue_text

    for wordgroup in root.iter("wordgroup"):
        kind = wordgroup.attrib.get("kind")
        if kind == "horizontal":
            direction = Direction.HORIZONTAL
        elif kind == "vertical":
            direction = Direction.VERTICAL
        else:
            continue

        for word in wordgroup.find("words").iter("word"):
            cells = word.find("cells").findall("cell")
            if not cells:
                continue
            start = Point(int(cells[0].attrib["y"]), int(cells[0].attrib["x"]))
            end = Point(int(cells[-1].attrib["y"]), int(cells[-1].attrib["x"]))

            w_idx = word.attrib["index"]
            clue_txt = clues[direction].get(w_idx, "")
            answ_node = word.find("puzzleword")
            answer = answ_node.text if answ_node is not None else ""

            clue_answer_pairs.append(
                ClueAnswerPair(clue_txt, answer, start, end, direction)
            )

    return Grid(
        layout=grid_layout,
        clue_answer_pairs=clue_answer_pairs,
        publisher="Le Figaro",
        publishing_date=date,
        difficulty=difficulty,
        source="keesing.com",
    )


# ==========================================
# 2. MULTI-THREADED SCRAPER
# ==========================================


def process_puzzle_id(puzzle_id: int):
    """
    Worker function.
    """
    filename = f"{puzzle_id}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 1. CHECK IF EXISTS (Skip if true)
    if os.path.exists(filepath):
        # Using tqdm.write prevents the progress bar from breaking
        # tqdm.write(f"[SKIP] {puzzle_id} (Exists)")
        return "SKIP"

    url = f"{BASE_URL}{puzzle_id}"

    try:
        # 2. REQUEST
        response = requests.get(url, timeout=10)
        response.encoding = "utf-8"

        if response.status_code != 200:
            return "HTTP_ERR"

        # 3. CONVERT
        grid = convert_lefigaro_arrow_crossword_grid(response.text)

        # 4. SAVE (UTF-8 Safe)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(grid.to_json())

        # Only log successes to keep output clean
        tqdm.write(f"[OK]   {puzzle_id}")
        return "OK"

    except GridConversionError:
        return "MISS"
    except Exception as e:
        tqdm.write(f"[ERR]  {puzzle_id}: {e}")
        return "ERR"


def generate_interleaved_ids(center: int, radius: int) -> Iterator[int]:
    yield center
    for i in range(1, radius + 1):
        yield center + i
        yield center - i


def run_scraper():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Calculate total for progress bar
    total_ids = (SEARCH_RADIUS * 2) + 1

    print(f"Starting scrape with {MAX_WORKERS} threads.")
    print(f"Target: {total_ids} IDs around {SEED_ID}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # 1. Submit all tasks
        futures = []
        for pid in generate_interleaved_ids(SEED_ID, SEARCH_RADIUS):
            futures.append(executor.submit(process_puzzle_id, pid))

        # 2. Process as they complete with Progress Bar
        # unit="grid" makes the bar say "100/5000 grids"
        for _ in tqdm(
            concurrent.futures.as_completed(futures), total=total_ids, unit="grid"
        ):
            pass

    print("Scraping finished.")


if __name__ == "__main__":
    run_scraper()
