import os
import json
import random
import pandas as pd
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Generator
from collections import defaultdict

# ==========================================
# 1. SPLITTING LOGIC
# ==========================================


def get_dataset_splits(
    json_dir: str,
    csv_dir: str,
    train_ratio: float = 0.79,
    val_ratio: float = 0.01,  # For the Neural Network (Early Stopping)
    calibration_ratio: float = 0.1,  # For the Solver (Hyperparameters)
    test_ratio: float = 0.1,  # For Final Reporting
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """
    Splits data into 4 sets: Train, Val, Calibration, Test.
    """

    # 1. Validate ratios
    total_ratio = train_ratio + val_ratio + calibration_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0. Got {total_ratio}")

    # 2. Gather files
    json_root = Path(json_dir)
    csv_root = Path(csv_dir)
    all_files = list(json_root.rglob("*.json")) + list(csv_root.rglob("*.csv"))

    if not all_files:
        return {"train": [], "val": [], "calibration": [], "test": []}

    # 3. Shuffle
    random.seed(seed)
    random.shuffle(all_files)

    # 4. Calculate Indices
    n = len(all_files)
    idx_train = int(n * train_ratio)
    idx_val = idx_train + int(n * val_ratio)
    idx_cal = idx_val + int(n * calibration_ratio)

    # 5. Split
    splits = {
        "train": all_files[:idx_train],
        "val": all_files[idx_train:idx_val],
        "calibration": all_files[idx_val:idx_cal],
        "test": all_files[idx_cal:],
    }

    print(f"Total: {n}")
    print(f"Train:       {len(splits['train'])} (Model Weights)")
    print(f"Validation:  {len(splits['val'])} (Model Early Stopping)")
    print(f"Calibration: {len(splits['calibration'])} (Solver Tuning)")
    print(f"Test:        {len(splits['test'])} (Final Metric)")

    return splits


# ==========================================
# 2. EXTRACTION LOGIC
# ==========================================


def _extract_from_json(filepath: Path) -> Generator[Tuple[str, str], None, None]:
    """Helper to parse a single JSON grid file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Determine format version
        # New Format: key is "clues", items have "clue" and "ans"
        # Old Format: key is "clue_answer_pairs", items have "clue" and "answer"

        raw_clues = []
        if "clues" in data:
            raw_clues = data["clues"]  # New format
        elif "clue_answer_pairs" in data:
            raw_clues = data["clue_answer_pairs"]  # Old format

        for item in raw_clues:
            # Handle key variations
            c_text = item.get("clue")
            a_text = item.get("ans", item.get("answer"))

            yield c_text, a_text

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Skipping corrupt JSON {filepath}: {e}")


def _extract_from_csv(filepath: Path) -> Generator[Tuple[str, str], None, None]:
    """Helper to parse a single CSV file."""
    try:
        # Pandas is robust for CSV parsing (handling quotes, newlines in fields)
        df = pd.read_csv(filepath)

        # Normalize headers to lowercase to find 'clue' and 'answer'
        df.columns = [c.lower().strip() for c in df.columns]

        if "clue" not in df.columns or "answer" not in df.columns:
            print(f"Warning: CSV {filepath} missing 'clue' or 'answer' columns.")
            return

        for _, row in df.iterrows():
            yield row["clue"], row["answer"]

    except Exception as e:
        print(f"Warning: Skipping corrupt CSV {filepath}: {e}")


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
        filter(str.isalpha, unicodedata.normalize("NFD", ans.upper()))
    ).replace("Œ", "OE")


def extract_pairs_from_split(file_list: List[Path]) -> Dict[str, List[str]]:
    """
    Iterates through a list of files, extracts clue-answer pairs,
    and filters out empty/invalid entries.

    Returns:
        A list of dicts: [(clue, answer), ...]
    """
    clues = []
    answers = []
    clue_answer_pairs = set()

    for filepath in file_list:

        # Select extractor based on extension
        extractor = None
        if filepath.suffix.lower() == ".json":
            extractor = _extract_from_json(filepath)
        elif filepath.suffix.lower() == ".csv":
            extractor = _extract_from_csv(filepath)
        else:
            continue

        # Iterate over the generator
        for clue, answer in extractor:

            # --- VALIDATION & CLEANING ---

            # 1. Check for None
            if clue is None or answer is None:
                continue

            # 2. Ensure strings
            if not isinstance(clue, str):
                clue = str(clue)
            if not isinstance(answer, str):
                answer = str(answer)

            # 3. Check for Empty / Whitespace only
            clue_clean = normalize_clue(clue)
            answer_clean = normalize_answer(answer)

            if not clue_clean:
                continue  # Discard empty clue

            # Optional: Discard empty answer? (Usually yes for training)
            if not answer_clean:
                continue
            if (clue_clean, answer_clean) not in clue_answer_pairs:
                clues.append(clue_clean)
                answers.append(answer_clean)
                clue_answer_pairs.add((clue_clean, answer_clean))

    return {"clue": clues, "answer": answers}


if __name__ == "__main__":
    json_dir = "data/grids"
    csv_dir = "data/list_clue_answers"
    output_dir = "data/train_test_val"

    splits = get_dataset_splits(json_dir, csv_dir, seed=123)

    for mode, split in splits.items():
        with open(output_dir + f"/{mode}_files.txt", "w", encoding="utf8") as f:
            f.write("\n".join(str(x) for x in split))
    for mode, split in splits.items():
        df = pd.DataFrame(extract_pairs_from_split(split))
        df.to_parquet(output_dir + f"/{mode}-00000-of-00001.parquet")
