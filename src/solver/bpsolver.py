"""
A solver based on loopy belief propagation. We more or less follow the procedure in https://www.aaai.org/Papers/AAAI/1999/AAAI99-023.pdf
where we make a variable for each fill (BPVar) and for each letter (BPCell).
BPVars connect to BPCells and vice versa in a connected graph, and we propagate alternating between the two,
being careful to preserve the directionality of the updates (e.g., the update from A to B should take into account A's neighbors other than B.)
The paper's procedure of infilling / finding words not in the original provided candidate list is not implemented.
We instead using iterative search after the fact by replacing characters one-by-one in our solution.
"""

import pandas as pd
import math
import string
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
from scipy.special import log_softmax, softmax
from tqdm import trange

from solver.utils import print_grid, get_word_flips
from solver.solver import Solver
from models import Reranker


# the probability of each alphabetical character in the crossword
UNIGRAM_PROBS = [
    ("A", 0.09028146545238917),
    ("B", 0.014518006177592508),
    ("C", 0.03505263786740302),
    ("D", 0.020129725017479578),
    ("E", 0.19116075458672968),
    ("F", 0.00896824027117684),
    ("G", 0.017465195727017675),
    ("H", 0.008331010098328156),
    ("I", 0.08466698085654356),
    ("J", 0.0010609439856977228),
    ("K", 0.0009453353866305569),
    ("L", 0.046664940834948536),
    ("M", 0.025458230447211676),
    ("N", 0.06676756144403438),
    ("O", 0.052534428130171965),
    ("P", 0.02305368221685297),
    ("Q", 0.001631242864349627),
    ("R", 0.09439691031870359),
    ("S", 0.08389035658338423),
    ("T", 0.07272555292993123),
    ("U", 0.04345389816707821),
    ("V", 0.011729017868996097),
    ("W", 0.00022623883740895132),
    ("X", 0.0024875209091150467),
    ("Y", 0.0016445184929506412),
    ("Z", 0.0007556045278743948),
]

# the LETTER_SMOOTHING_FACTOR controls how much we interpolate with the unigram LM. TODO this should be tuned.
# Right now it is set according to the probability that the answer is not in the answer set
LETTER_SMOOTHING_FACTOR = [
    0.0,
    0.0,
    0.04395604395604396,
    0.0001372495196266813,
    0.0005752186417796561,
    0.0019841824329989103,
    0.0048042463338563764,
    0.013325257419745608,
    0.027154447774285505,
    0.06513517299341645,
    0.12527790128946198,
    0.22003002358996354,
    0.23172376584839494,
    0.254873006497342,
    0.3985086992543496,
    0.2764976958525346,
    0.672645739910314,
    0.6818181818181818,
    0.8571428571428571,
    0.8245614035087719,
    0.8,
    0.71900826446281,
    0.0,
]

MIN_PROBABILITY = 0.01
MIN_SCORE_DIFF = 0.05


class BPVar:
    def __init__(self, name, variable, candidates, cells):
        """
        Représente une **variable de mot** dans un puzzle de mots croisés pour la propagation de croyances (belief propagation).

        Chaque instance correspond à un mot spécifique de la grille et gère :
            - Les lettres du mot solution (`gold word`).
            - Les candidats possibles et leurs probabilités.
            - Les scores locaux et directionnels pour la propagation.
            - La communication avec les cellules qu il occupe pour mettre à jour les croyances sur les lettres.

        Args:
            name (str):  nom du mot.
            variable (dict): Contient les informations du mot, incluant la liste de ses cellules et les indices de mots croisés intersectants.
            candidates (dict): Dictionnaire avec les mots candidats, leurs poids (coûts), et la représentation binaire de chaque lettre.
            cells (list of BPCell): Liste des cellules correspondant aux lettres du mot dans la grille.

        Attributs principaux:
            self.name (str): Nom du mot.
            self.length (int): Nombre de lettres du mot.
            self.ordered_cells (list of BPCell): Cellules de la grille dans l ordre du mot.
            self.candidates (dict): Dictionnaire des candidats possibles.
            self.words (list of str): Liste des mots candidats.
            self.word_indices (np.array): Représentation numérique des lettres des candidats.
            self.scores (np.array): Scores négatifs (coûts) des mots candidats.
            self.prior_log_probs (np.array): Probabilités a priori log-normalisées.
            self.log_probs (np.array): Probabilités log-normalisées mises à jour après propagation.
            self.directional_scores (list of np.array): Scores venant de chaque cellule pour la propagation.
        """
        self.name = name
        cells_by_position = {}
        for cell in cells:
            cells_by_position[cell.position] = cell
            cell._connect(self)
        self.length = len(cells)
        self.ordered_cells = [cells_by_position[pos] for pos in variable["cells"]]
        self.candidates = candidates
        self.words = self.candidates["words"]
        self.word_indices = np.array(
            [
                [string.ascii_uppercase.index(l) for l in fill]
                for fill in self.candidates["words"]
            ]
        )  # words x length of letter indices
        self.scores = -np.array(
            [self.candidates["weights"][fill] for fill in self.candidates["words"]]
        )  # the incoming 'weights' are costs
        self.prior_log_probs = log_softmax(self.scores)
        self.log_probs = log_softmax(self.scores)
        self.directional_scores = [
            np.zeros(len(self.log_probs)) for _ in range(len(self.ordered_cells))
        ]

    def _propagate_to_var(self, other, belief_state):
        assert other in self.ordered_cells
        other_idx = self.ordered_cells.index(other)
        letter_scores = belief_state
        self.directional_scores[other_idx] = letter_scores[
            self.word_indices[:, other_idx]
        ]

    def _postprocess(self, all_letter_probs):
        # unigram smoothing
        unigram_probs = np.array([x[1] for x in UNIGRAM_PROBS])
        for i in range(len(all_letter_probs)):
            all_letter_probs[i] = (
                1 - LETTER_SMOOTHING_FACTOR[self.length]
            ) * all_letter_probs[i] + LETTER_SMOOTHING_FACTOR[
                self.length
            ] * unigram_probs
        return all_letter_probs

    def sync_state(self):
        self.log_probs = log_softmax(
            sum(self.directional_scores) + self.prior_log_probs
        )

    def propagate(self):
        all_letter_probs = []
        for i in range(len(self.ordered_cells)):
            word_scores = self.log_probs - self.directional_scores[i]
            word_probs = softmax(word_scores)
            letter_probs = (
                self.candidates["bit_array"][:, i] * np.expand_dims(word_probs, axis=0)
            ).sum(axis=1) + 1e-8
            all_letter_probs.append(letter_probs)
        all_letter_probs = self._postprocess(all_letter_probs)  # unigram postprocessing
        for i, cell in enumerate(self.ordered_cells):
            cell._propagate_to_cell(self, np.log(all_letter_probs[i]))


class BPCell:

    def __init__(self, position, clue_pair):
        """
        Représente une **cellule de lettre** dans la grille de mots croisés pour la propagation de croyances.

        Chaque cellule peut être traversée par jusqu’à deux mots (variables). La cellule :
            - Maintient les probabilités log pour chaque lettre.
            - Stocke les variables qui la traversent (crossing_vars) et leurs scores directionnels.
            - Permet la propagation des croyances entre lettres et mots.

        Args:
            position (tuple): Coordonnées (ligne, colonne) de la cellule dans la grille.
            clue_pair (list of str): Liste des indices de mots croisés qui passent par cette cellule (ex : ["1A", "5D"]).

        Attributs principaux:
            self.position (tuple): Position de la cellule dans la grille.
            self.crossing_clues (list of str): Clues qui passent par cette cellule.
            self.letters (list of str): Alphabet complet pour les probabilités.
            self.log_probs (np.array): Probabilités log pour chaque lettre.
            self.crossing_vars (list of BPVar): Variables (mots) traversant cette cellule.
            self.directional_scores (list of np.array): Scores directionnels venant des variables.
            self.prediction (dict): Mot prédit pour chaque clue associé à cette cellule.
        """
        self.crossing_clues = clue_pair
        self.position = tuple(position)
        self.letters = list(string.ascii_uppercase)
        self.log_probs = np.log(
            np.array([1.0 / len(self.letters) for _ in range(len(self.letters))])
        )
        self.crossing_vars = []
        self.directional_scores = []
        self.prediction = {}

    def _connect(self, other):
        self.crossing_vars.append(other)
        self.directional_scores.append(None)
        assert len(self.crossing_vars) <= 2

    def _propagate_to_cell(self, other, belief_state):
        assert other in self.crossing_vars
        other_idx = self.crossing_vars.index(other)
        self.directional_scores[other_idx] = belief_state

    def sync_state(self):
        self.log_probs = log_softmax(sum(self.directional_scores))

    def propagate(self):
        assert len(self.crossing_vars) == 2
        for i, v in enumerate(self.crossing_vars):
            v._propagate_to_var(self, self.directional_scores[1 - i])


class BPSolver(Solver):
    def __init__(
        self, crossword, retriever, encoder, reranker, max_candidates, **kwargs
    ):
        super().__init__(crossword, retriever, encoder, max_candidates)
        self.crossword = crossword
        self.reranker = reranker
        self.reset()

    def reset(self):
        self.bp_cells = []
        self.bp_cells_by_clue = defaultdict(lambda: [])
        for position, clue_pair in self.crossword.grid_cells.items():
            cell = BPCell(position, clue_pair)
            self.bp_cells.append(cell)
            for clue in clue_pair:
                self.bp_cells_by_clue[clue].append(cell)
        self.bp_vars = []
        for key, value in self.crossword.variables.items():
            var = BPVar(key, value, self.candidates[key], self.bp_cells_by_clue[key])
            self.bp_vars.append(var)

    def solve(
        self,
        num_iters=10,
        iterative_improvement_steps=5,
        return_greedy_states=False,
        return_ii_states=False,
    ):
        # run solving for num_iters iterations
        print("beginning BP iterations")
        for _ in trange(num_iters):
            for var in self.bp_vars:
                var.propagate()
            for cell in self.bp_cells:
                cell.sync_state()
            for cell in self.bp_cells:
                cell.propagate()
            for var in self.bp_vars:
                var.sync_state()
        print("done BP iterations")

        # Get the current based grid based on greedy selection from the marginals
        if return_greedy_states:
            grid, all_grids = self.greedy_sequential_word_solution(return_grids=True)
        else:
            grid = self.greedy_sequential_word_solution()
            all_grids = []
        grid = self.greedy_sequential_word_solution()
        print("=====Greedy search grid=====")
        print_grid(grid)

        if iterative_improvement_steps < 1:
            if return_greedy_states or return_ii_states:
                return grid, all_grids
            else:
                return grid

        for i in range(iterative_improvement_steps):
            print("starting iterative improvement step " + str(i))
            print("Accuracy if we knew to stop right now")
            self.evaluate(grid)
            grid, did_iterative_improvement_make_edit = self.iterative_improvement(grid)
            if not did_iterative_improvement_make_edit:
                break
            if return_ii_states:
                all_grids.append(deepcopy(grid))
            print("after iterative improvement step " + str(i))
            print_grid(grid)

        if return_greedy_states or return_ii_states:
            return grid, all_grids
        else:
            return grid

    def get_candidate_replacements(self, uncertain_answers, grid):
        # find alternate answers for all the uncertain words
        candidate_replacements = []
        replacement_id_set = set()

        # check against dictionaries
        for clue in uncertain_answers.keys():
            initial_word = uncertain_answers[clue]
            clue_flips = get_word_flips(initial_word, 10)  # flip then segment
            clue_positions = [
                key
                for key, value in self.crossword.variables.items()
                if value["clue"] == clue
            ]
            for clue_position in clue_positions:
                cells = sorted(
                    [
                        cell
                        for cell in self.bp_cells
                        if clue_position in cell.crossing_clues
                    ],
                    key=lambda c: c.position,
                )
                if len(cells) == len(initial_word):
                    break
            for flip in clue_flips:
                if len(flip) != len(cells):
                    import pdb

                    pdb.set_trace()
                assert len(flip) == len(cells)
                for i in range(len(flip)):
                    if flip[i] != initial_word[i]:
                        candidate_replacements.append(
                            [(cells[i], flip[i])]
                        )  # c'est une liste de liste d'un tuple (bpcellule et une lettre )
                        break

        # also add candidates based on uncertainties in the letters, e.g., if we said P but G also had some probability, try G too
        for cell_id, cell in enumerate(self.bp_cells):
            probs = np.exp(cell.log_probs)
            above_threshold = list(probs > MIN_PROBABILITY)
            new_characters = [
                string.ascii_uppercase[i] for i in range(26) if above_threshold[i]
            ]
            # used = set()
            # new_characters = [x for x in new_characters if x not in used and (used.add(x) or True)] # unique the set
            new_characters = [
                x
                for x in new_characters
                if x != grid[cell.position[0]][cell.position[1]]
            ]  # ignore if its the same as the original solution
            if len(new_characters) > 0:
                for new_character in new_characters:
                    id = "_".join([str(cell.position), new_character])
                    if id not in replacement_id_set:
                        candidate_replacements.append([(cell, new_character)])
                    replacement_id_set.add(id)

        # create composite flips based on things in the same row/column. chaque flip = modif 1 lettre. maintenant on autorise modif de 2 lettres dans un meme mot
        composite_replacements = []
        for i in range(len(candidate_replacements)):
            for j in range(i + 1, len(candidate_replacements)):
                flip1, flip2 = candidate_replacements[i], candidate_replacements[j]
                if flip1[0][0] != flip2[0][0]:
                    if (
                        len(
                            set(flip1[0][0].crossing_clues + flip2[0][0].crossing_clues)
                        )
                        < 4
                    ):  # shared clue
                        composite_replacements.append(flip1 + flip2)

        candidate_replacements += composite_replacements

        # print('\ncandidate replacements') (juste affichage)
        for cr in candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in cr:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            variables = set(
                sum([cell.crossing_vars for cell, _ in cr], [])
            )  #  variables toucher par les modifs
            for var in variables:
                original_fill = "".join(
                    [
                        grid[cell.position[0]][cell.position[1]]
                        for cell in var.ordered_cells
                    ]
                )
                modified_fill = "".join(
                    [
                        modified_grid[cell.position[0]][cell.position[1]]
                        for cell in var.ordered_cells
                    ]
                )
                # print('original:', original_fill, 'modified:', modified_fill)

        return candidate_replacements

    def get_uncertain_answers(self, grid):
        original_qa_pairs = {}  # the original puzzle preds that we will try to improve
        # first save what the argmax word-level prediction was for each grid cell just to make life easier
        for var in self.crossword.variables:
            # read the current word off the grid
            cells = self.crossword.variables[var]["cells"]
            word = []
            for cell in cells:
                word.append(grid[cell[0]][cell[1]])
            word = "".join(word)
            for cell in self.bp_cells:  # loop through all cells
                if (
                    cell.position in cells
                ):  # if this cell is in the word we are currently handling
                    # save {clue, answer} pair into this cell
                    cell.prediction[self.crossword.variables[var]["clue"]] = word
                    original_qa_pairs[self.crossword.variables[var]["clue"]] = word

        uncertain_answers = {}

        # find uncertain answers
        # right now the heuristic we use is any answer that is not in the answer set
        for clue in original_qa_pairs.keys():
            if original_qa_pairs[clue] not in answer_set:
                uncertain_answers[clue] = original_qa_pairs[clue]

        return uncertain_answers

    def score_grid(self, grid):
        clues = []
        answers = []
        for clue, cells in self.bp_cells_by_clue.items():
            letters = "".join(
                [
                    grid[cell.position[0]][cell.position[1]]
                    for cell in sorted(list(cells), key=lambda c: c.position)
                ]
            )
            clues.append(self.crossword.variables[clue]["clue"])
            answers.append(letters)
        scores = self.reranker(clues, answers)
        return sum(scores)

    def greedy_sequential_word_solution(self, return_grids=False):
        all_grids = []
        # after we've run BP, we run a simple greedy search to get the final.
        # We repeatedly pick the highest-log-prob candidate across all clues which fits the grid, and fill it.
        # at the end, if you have any cells left (due to missing gold candidates) just fill it with the argmax on that letter.
        cache = [(deepcopy(var.words), deepcopy(var.log_probs)) for var in self.bp_vars]

        grid = [["" for _ in row] for row in self.crossword.letter_grid]
        unfilled_cells = set([cell.position for cell in self.bp_cells])
        for var in self.bp_vars:
            # postprocess log probs to estimate probability that you don't have the right candidate
            var.log_probs = var.log_probs + math.log(
                1 - LETTER_SMOOTHING_FACTOR[var.length]
            )
        best_per_var = [var.log_probs.max() for var in self.bp_vars]
        while not all([x is None for x in best_per_var]):
            all_grids.append(deepcopy(grid))
            best_index = best_per_var.index(
                max([x for x in best_per_var if x is not None])
            )
            best_var = self.bp_vars[best_index]
            best_word = best_var.words[best_var.log_probs.argmax()]
            print("greedy filling in", best_word)
            for i, cell in enumerate(best_var.ordered_cells):
                letter = best_word[i]
                grid[cell.position[0]][cell.position[1]] = letter
                if cell.position in unfilled_cells:
                    unfilled_cells.remove(cell.position)
                for var in cell.crossing_vars:
                    if var != best_var:
                        cell_index = var.ordered_cells.index(cell)
                        keep_indices = [
                            j
                            for j in range(len(var.words))
                            if var.words[j][cell_index] == letter
                        ]  # on garde que les mots qui ont le bon indices pour les mots qui croisent le mot qu'on vient de placer
                        var.words = [var.words[j] for j in keep_indices]
                        var.log_probs = var.log_probs[keep_indices]
                        var_index = self.bp_vars.index(var)
                        if len(keep_indices) > 0:
                            best_per_var[var_index] = var.log_probs.max()
                        else:
                            best_per_var[var_index] = None
            best_var.words = []
            best_var.log_probs = best_var.log_probs[[]]
            best_per_var[best_index] = None
        for cell in self.bp_cells:
            if cell.position in unfilled_cells:
                grid[cell.position[0]][cell.position[1]] = string.ascii_uppercase[
                    cell.log_probs.argmax()
                ]

        for var, (words, log_probs) in zip(self.bp_vars, cache):  # restore state
            var.words = words
            var.log_probs = log_probs
        if return_grids:
            return grid, all_grids
        else:
            return grid

    def iterative_improvement(self, grid):
        # check the grid for uncertain areas and save those words to be analyzed in local search, aka looking for alternate candidates
        uncertain_answers = self.get_uncertain_answers(grid)
        self.candidate_replacements = self.get_candidate_replacements(
            uncertain_answers, grid
        )
        # on cherche les candidats qui modifient significativemment la grille
        print("\nstarting iterative improvement")
        original_grid_score = self.score_grid(grid)
        possible_edits = []
        for replacements in self.candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in replacements:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            modified_grid_score = self.score_grid(modified_grid)
            print("candidate edit")
            variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
            for var in variables:
                original_fill = "".join(
                    [
                        grid[cell.position[0]][cell.position[1]]
                        for cell in var.ordered_cells
                    ]
                )
                modified_fill = "".join(
                    [
                        modified_grid[cell.position[0]][cell.position[1]]
                        for cell in var.ordered_cells
                    ]
                )
                clue_index = list(
                    set(var.ordered_cells[0].crossing_clues).intersection(
                        *[set(cell.crossing_clues) for cell in var.ordered_cells]
                    )
                )[0]
                print("original:", original_fill, "modified:", modified_fill)
                print("gold answer", self.crossword.variables[clue_index]["gold"])
                print("clue", self.crossword.variables[clue_index]["clue"])
            print(
                "original score:",
                original_grid_score,
                "modified score:",
                modified_grid_score,
            )
            if modified_grid_score - original_grid_score > MIN_SCORE_DIFF:
                print("found a possible edit")
                possible_edits.append(
                    (modified_grid, modified_grid_score, replacements)
                )
            print()
        # on prend les meileurs edit possible sans qu'aucunes ne se croisent
        if len(possible_edits) > 0:
            variables_modified = set()
            possible_edits = sorted(possible_edits, key=lambda x: x[1], reverse=True)
            selected_edits = []
            for edit in possible_edits:
                replacements = edit[2]
                variables = set(
                    sum([cell.crossing_vars for cell, _ in replacements], [])
                )
                if (
                    len(variables_modified.intersection(variables)) == 0
                ):  # we can do multiple updates at once if they don't share clues
                    variables_modified.update(variables)
                    selected_edits.append(edit)
            # on realise les meileurs edit selectionner
            new_grid = deepcopy(grid)
            for edit in selected_edits:
                print("\nactually applying edit")
                replacements = edit[2]
                for cell, letter in replacements:
                    new_grid[cell.position[0]][cell.position[1]] = letter
                variables = set(
                    sum([cell.crossing_vars for cell, _ in replacements], [])
                )
                for var in variables:
                    original_fill = "".join(
                        [
                            grid[cell.position[0]][cell.position[1]]
                            for cell in var.ordered_cells
                        ]
                    )
                    modified_fill = "".join(
                        [
                            new_grid[cell.position[0]][cell.position[1]]
                            for cell in var.ordered_cells
                        ]
                    )
                    print("original:", original_fill, "modified:", modified_fill)
            return new_grid, True
        else:
            return grid, False
