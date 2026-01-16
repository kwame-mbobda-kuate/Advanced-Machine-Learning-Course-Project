from enum import Enum
from typing import Optional, List, Any, Dict
import json
import datetime
from scrapping.utils import normalize_clue, normalize_answer


class GridFilling(Enum):
    FREE = 0
    BLOCKED = 1

    @classmethod
    def from_char(cls, char: str):
        # Support old (#) and new (@) blocked characters
        if char in ["#", "@"]:
            return cls.BLOCKED
        return cls.FREE

    def to_char(self):
        # New visual style: @ for block, Space for free
        return "@" if self == self.BLOCKED else " "


class Direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

    def to_code(self):
        return "H" if self == self.HORIZONTAL else "V"

    @classmethod
    def from_code(cls, code: str):
        return cls.HORIZONTAL if code == "H" else cls.VERTICAL


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def to_list(self):
        return [self.x, self.y]

    @staticmethod
    def from_list(data: List[int]):
        return Point(data[0], data[1])


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
        return {
            "d": self.direction.to_code(),
            "xy": self.start.to_list(),
            "len": len(self.answer),
            "clue": self.clue,
            "ans": self.answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        start = Point.from_list(data["xy"])
        direction = Direction.from_code(data["d"])
        length = data.get("len", len(data["ans"]))

        if direction == Direction.HORIZONTAL:
            end = Point(start.x + length - 1, start.y)
        else:
            end = Point(start.x, start.y + length - 1)

        return cls(data["clue"], data["ans"], start, end, direction)


class Grid:
    def __init__(
        self,
        layout: List[List[GridFilling]],
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
        self.layout = layout
        self.clue_answer_pairs = clue_answer_pairs

    def to_json(self, indent=2):
        # 1. Layout to Visual Strings
        layout_visual = []
        for row in self.layout:
            row_str = "".join([cell.to_char() for cell in row])
            layout_visual.append(row_str)

        # 2. Clues
        clues_data = [c.to_dict() for c in self.clue_answer_pairs]

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
            # RENAMED FIELD
            "clue_answer_pairs": clues_data,
        }
        # ensure_ascii=False ensures UTF-8 chars like é are written literally, not as \u00e9
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str):
        # Handle double-encoded strings
        if isinstance(json_str, str):
            try:
                data = json.loads(json_str)
                if isinstance(data, str):
                    data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        else:
            data = json_str

        meta = data.get("meta", {})

        # Parse Layout
        layout = []
        for row_str in data.get("layout", []):
            grid_row = [GridFilling.from_char(char) for char in row_str]
            layout.append(grid_row)

        # COMPATIBILITY: Look for "clue_answer_pairs", fallback to "clues"
        raw_clues = data.get("clue_answer_pairs")
        if raw_clues is None:
            raw_clues = data.get("clues", [])

        clue_pairs = [ClueAnswerPair.from_dict(c) for c in raw_clues]

        p_date = meta.get("date")
        if p_date:
            p_date = datetime.date.fromisoformat(p_date)

        return cls(
            layout=layout,
            clue_answer_pairs=clue_pairs,
            publisher=meta.get("publisher"),
            publishing_date=p_date,
            author=meta.get("author"),
            source=meta.get("source"),
            difficulty=meta.get("difficulty"),
            title=meta.get("title"),
        )
    
    def normalize(self):
        for ca in self.clue_answer_pairs:
            ca.clue = normalize_clue(ca.clue)
            ca.answer = normalize_answer(ca.answer)
            


class GridConversionError(Exception):
    pass
