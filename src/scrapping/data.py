from enum import Enum
from typing import Optional, List, Any, Dict
import json
import datetime


class GridFilling(Enum):
    FREE = 0
    BLOCKED = 1

    @classmethod
    def from_char(cls, char: str):
        return cls.BLOCKED if char == "#" else cls.FREE

    def to_char(self):
        return "#" if self == self.BLOCKED else "."


class Direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

    def to_code(self):
        return "H" if self == self.HORIZONTAL else "V"

    @classmethod
    def from_code(cls, code: str):
        return cls.HORIZONTAL if code == "H" else cls.VERTICAL


class Point:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x

    def to_list(self):
        return [self.y, self.x]

    @staticmethod
    def from_list(data: List[int]):
        return Point(data[0], data[1])

    def __repr__(self):
        return f"({self.y},{self.x})"


class ClueAnswerPair:
    def __init__(
        self, clue: str, answer: str, start: Point, end: Point, direction: Direction
    ):
        self.clue = clue
        self.answer = answer
        self.start = start
        self.end = end
        self.direction = direction

    def to_dict(self):
        # We flatten the structure for readability.
        # 'end' is technically redundant if we have start+len+direction,
        # but we keep it for validation if you prefer.
        return {
            "d": self.direction.to_code(),
            "xy": self.start.to_list(),
            "len": len(
                self.answer
            ),  # Easier for humans to scan length than calc end coords
            "clue": self.clue,
            "ans": self.answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        start = Point.from_list(data["xy"])
        direction = Direction.from_code(data["d"])

        # Reconstruct End point based on length and direction
        length = data.get("len", len(data["ans"]))
        if direction == Direction.HORIZONTAL:
            end = Point(start.y, start.x + length - 1)
        else:
            end = Point(start.y + length - 1, start.x)

        return cls(
            clue=data["clue"],
            answer=data["ans"],
            start=start,
            end=end,
            direction=direction,
        )

    def __repr__(self):
        return f"{self.clue} -> {self.answer}"


class Grid:
    def __init__(
        self,
        grid_layout: List[List[GridFilling]],
        clue_answer_pairs: List[ClueAnswerPair],
        publisher: Optional[str] = None,
        publishing_date: Optional[datetime.date] = None,
        author: Optional[str] = None,
        source: Optional[str] = None,
        difficulty: Optional[int] = None,
        title: Optional[str] = None,
    ):
        self.publisher = publisher
        self.publishing_date = publishing_date
        self.author = author
        self.source = source
        self.difficulty = difficulty
        self.title = title
        self.grid_layout = grid_layout
        self.clue_answer_pairs = clue_answer_pairs
        self.title = title

    def to_json(self, indent=2):
        """Produces a human-readable JSON string."""

        # 1. Convert Layout to Visual ASCII
        layout_visual = []
        for row in self.grid_layout:
            row_str = "".join([cell.to_char() for cell in row])
            layout_visual.append(row_str)

        # 2. Convert Clues
        clues_data = [c.to_dict() for c in self.clue_answer_pairs]

        # 3. Build Final Dict
        data = {
            "meta": {
                "title": self.title,
                "author": self.author,
                "publisher": self.publisher,
                "date": (
                    self.publishing_date.isoformat() if self.publishing_date else None
                ),
                "difficulty": self.difficulty,
                "source": self.source,
            },
            "layout": layout_visual,
            "clues": clues_data,
        }
        return json.dumps(data, indent=indent)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        meta = data.get("meta", {})

        # 1. Parse Layout
        grid_layout = []
        for row_str in data["layout"]:
            grid_row = [GridFilling.from_char(char) for char in row_str]
            grid_layout.append(grid_row)

        # 2. Parse Clues
        clue_pairs = [ClueAnswerPair.from_dict(c) for c in data["clues"]]

        # 3. Parse Date
        p_date = meta.get("date")
        if p_date:
            p_date = datetime.date.fromisoformat(p_date)

        return cls(
            grid_layout=grid_layout,
            clue_answer_pairs=clue_pairs,
            publisher=meta.get("publisher"),
            publishing_date=p_date,
            author=meta.get("author"),
            source=meta.get("source"),
            difficulty=meta.get("difficulty"),
            title=meta.get("title"),
        )
