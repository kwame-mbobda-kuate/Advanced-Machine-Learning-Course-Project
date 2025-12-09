from enum import Enum
from typing import Optional, List, Tuple, Any
import json
import datetime


class GridFilling(Enum):
    FREE = 0
    BLOCKED = 1


class Direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Point:

    def __init__(self, y: int, x: int):
        self.x = x
        self.y = y


class ClueAnswerPair:

    def __init__(
        self, clue: str, answer: str, start: Point, end: Point, direction: Direction
    ):
        self.clue = clue
        self.answer = answer
        self.start = start
        self.end = end
        self.direction = direction

    def __repr__(self):
        return f"{self.clue} -> {self.answer}"


class Grid:

    def __init__(
        self,
        grid_layout: List[List[GridFilling]],
        clue_answer_pairs: List[ClueAnswerPair],
        publisher: Optional[str] = None,
        publishing_date: datetime.date = None,
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
        self.grid_layout = grid_layout
        self.clue_answer_pairs = clue_answer_pairs


class GridConversionError(Exception):
    pass


class GridEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, GridFilling):
            return {"__enum__": "GridFilling", "value": obj.value}
        elif isinstance(obj, Direction):
            return {"__enum__": "Direction", "value": obj.value}
        elif isinstance(obj, Point):
            return {"__type__": "Point", "x": obj.x, "y": obj.y}
        elif isinstance(obj, ClueAnswerPair):
            return {
                "__type__": "ClueAnswerPair",
                "clue": obj.clue,
                "answer": obj.answer,
                "start": obj.start,
                "end": obj.end,
                "direction": obj.direction,
            }
        elif isinstance(obj, datetime.date):
            return {"__type__": "date", "value": obj.isoformat()}
        elif isinstance(obj, Grid):
            return {
                "__type__": "Grid",
                "publisher": obj.publisher,
                "publishing_date": obj.publishing_date,
                "author": obj.author,
                "source": obj.source,
                "difficulty": obj.difficulty,
                "grid_layout": obj.grid_layout,
                "clue_answer_pairs": obj.clue_answer_pairs,
            }
        return super().default(obj)


def grid_decoder(dct: dict) -> Any:
    if "__enum__" in dct:
        if dct["__enum__"] == "GridFilling":
            return GridFilling(dct["value"])
        elif dct["__enum__"] == "Direction":
            return Direction(dct["value"])
    if "__type__" in dct:
        if dct["__type__"] == "Point":
            return Point(dct["y"], dct["x"])
        elif dct["__type__"] == "ClueAnswerPair":
            return ClueAnswerPair(
                dct["clue"],
                dct["answer"],
                dct["start"],
                dct["end"],
                dct["direction"],
            )
        elif dct["__type__"] == "date":
            return datetime.date.fromisoformat(dct["value"])
        elif dct["__type__"] == "Grid":
            return Grid(
                dct["grid_layout"],
                dct["clue_answer_pairs"],
                dct["publisher"],
                dct["publishing_date"],
                dct["author"],
                dct["source"],
                dct["difficulty"],
            )
    return dct
