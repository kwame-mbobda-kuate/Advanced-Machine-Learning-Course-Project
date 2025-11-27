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