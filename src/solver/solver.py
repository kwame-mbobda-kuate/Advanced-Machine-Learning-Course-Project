from collections import defaultdict
import string
from scipy.special import softmax
import numpy as np
from models import Retriever, Encoder
from crossword import Crossword


class Solver:
    """
    This class represents an abstraction over different types of crossword solvers. Each puzzle contains
    a list of clues, which are associated with (weighted) values for each candidate answer.

    Args:
        crossword (Crossword): puzzle to solve
        max_candidates (int): number of answer candidates to consider per clue
    """

    def __init__(
        self,
        crossword: Crossword,
        retriever: Retriever,
        encoder: Encoder,
        max_candidates: int,
    ):
        self.crossword = crossword
        self.retriever = retriever
        self.encoder = encoder
        self.max_candidates = max_candidates
        self.get_candidates()

    def get_candidates(self):
        # get answers from neural model and fill up data structures with the results
        chars = string.ascii_uppercase
        self.char_map = {char: idx for idx, char in enumerate(chars)}
        self.candidates = {}

        all_clues = []
        for var in self.crossword.variables:
            all_clues.append(self.crossword.variables[var]["clue"])

        # get predictions
        encodings = self.encoder.encode(all_clues)
        all_words, all_scores = self.retriever.retrieve(encodings, self.max_candidates)
        for index, var in enumerate(self.crossword.variables):
            length = len(self.crossword.variables[var]["gold"])
            self.candidates[var] = {"words": [], "bit_array": None, "weights": {}}

            clue = all_clues[index]
            words, scores = all_words[index], all_scores[index]
            # remove answers that are not of the correct length
            keep_positions = []
            for word_index, word in enumerate(words):
                if len(word) == length:
                    keep_positions.append(word_index)
            words = [words[i] for i in keep_positions]
            scores = [scores[i] for i in keep_positions]
            scores = list(-np.log(softmax(np.array(scores) / 0.75)))

            for word, score in zip(words, scores):
                self.candidates[var]["weights"][word] = score

            # for debugging purposes, print the rank of the gold answer on our candidate list
            # the gold answer is otherwise *not* used in any way during solving
            if self.crossword.variables[var]["gold"] in words:
                print(
                    clue,
                    self.crossword.variables[var]["gold"],
                    words.index(self.crossword.variables[var]["gold"]),
                )
            else:
                print("not found", clue, self.crossword.variables[var]["gold"])

            # fill up some data structures used later in solving
            for word, score in zip(words, scores):
                self.candidates[var]["weights"][word] = score
            weights = self.candidates[var]["weights"]
            self.candidates[var]["words"] = sorted(weights, key=weights.get)
            self.candidates[var]["bit_array"] = np.zeros(
                (len(chars), length, len(self.candidates[var]["words"]))
            )
            self.candidates[var]["single_query_cache"] = [
                defaultdict(lambda: []) for _ in range(len(chars))
            ]
            self.candidates[var]["single_query_cache_indices"] = [
                defaultdict(lambda: []) for _ in range(len(chars))
            ]
            for word_idx, word in enumerate(self.candidates[var]["words"]):
                for pos_idx, char in enumerate(word):
                    char_idx = self.char_map[char]
                    self.candidates[var]["bit_array"][char_idx, pos_idx, word_idx] = 1
                    self.candidates[var]["single_query_cache"][pos_idx][char].append(
                        word
                    )
                    self.candidates[var]["single_query_cache_indices"][pos_idx][
                        char
                    ].append(word_idx)
                    # NOTE: TODO, it's possible to cache more here in exchange for doing more work at init time

    def evaluate(self, solution):
        # calculate puzzle accuracy results given a generated solution
        letters_correct = 0
        letters_total = 0
        for i in range(len(self.crossword.letter_grid)):
            for j in range(len(self.crossword.letter_grid[0])):
                if self.crossword.letter_grid[i][j] != "":
                    letters_correct += (
                        self.crossword.letter_grid[i][j] == solution[i][j]
                    )
                    letters_total += 1

        words_correct = 0
        words_total = 0
        for var in self.crossword.variables:
            cells = self.crossword.variables[var]["cells"]
            matching_cells = [
                self.crossword.letter_grid[cell[0]][cell[1]]
                == solution[cell[0]][cell[1]]
                for cell in cells
            ]
            if len(cells) == sum(matching_cells):
                words_correct += 1
            words_total += 1

        letter_accuracy = letters_correct / letters_total if letters_total > 0 else 0.0
        word_accuracy = words_correct / words_total if words_total > 0 else 0.0

        return {
            "letters_correct": letters_correct,
            "letters_total": letters_total,
            "words_correct": words_correct,
            "words_total": words_total,
            "letter_accuracy": letter_accuracy,
            "word_accuracy": word_accuracy,
        }
