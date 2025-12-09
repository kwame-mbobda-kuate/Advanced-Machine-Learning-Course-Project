from data import Grid, GridFilling, Direction, ClueAnswerPair, Point, GridConversionError
import datetime
import xml.etree.ElementTree as ET
import json


def convert_lefigaro_arrow_crossword_grid(crossword_grid: str) -> Grid:
    root = ET.fromstring(crossword_grid)
    clues = {Direction.HORIZONTAL: {}, Direction.VERTICAL: {}}
    grid_layout = [[GridFilling.BLOCKED] * int(root.attrib["width"]) for _ in range(int(root.attrib["height"]))]
    difficulty = int(root.attrib["difficulty"])
    date = datetime.datetime.fromisoformat(root.attrib["exported"])
    clue_answer_pairs = []

    for cell in root.find("grid").iter("cell"):
        x, y = int(cell.attrib["x"]), int(cell.attrib["y"])
        if cell.attrib["fillable"] == "1":
            grid_layout[y][x] = GridFilling.FREE
        for clue in cell.iter("clue"):
            clue_text = clue.text.replace("\\", "")
            if clue.attrib["arrow"] in ["arrowdownrightbottom", "arrowrighttop", "arrowdownright", "arrowright"]:
                direction = Direction.HORIZONTAL
            elif clue.attrib["arrow"] in ["arrowdownbottom", "arrowrightdowntop", "arrowdown"]:
                direction = Direction.VERTICAL
            else:
                raise GridConversionError(f"Unknown value for the 'arrow' field: {clue.attrib['arrow']}.")
            clues[direction][clue.attrib["wordindex"]] = clue_text

    for wordgroup in root.iter("wordgroup"):
        if wordgroup.attrib["kind"] == "horizontal":
            direction = Direction.HORIZONTAL
        if wordgroup.attrib["kind"] == "vertical":
            direction = Direction.VERTICAL
        for word in wordgroup.find("words").iter("word"):
            cells = word.find("cells").findall("cell")
            start = Point(int(cells[0].attrib["y"]), int(cells[0].attrib["x"]))
            end = Point(int(cells[-1].attrib["y"]), int(cells[-1].attrib["x"]))
            clue_answer_pair = ClueAnswerPair(clues[direction][word.attrib["index"]], word.find("puzzleword").text, start, end, direction)
            clue_answer_pairs.append(clue_answer_pair)
    return Grid(grid_layout=grid_layout, clue_answer_pairs=clue_answer_pairs, publisher="Le Figaro", publishing_date=date, difficulty=difficulty)


if __name__ == "__main__":
    with open("data/test/lefigaro_grid_1.xml", "r") as f:
        grid = convert_lefigaro_arrow_crossword_grid(f.read())
        print(json.dumps())
