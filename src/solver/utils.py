import math
from wordfreq import word_frequency
from collections import defaultdict

PENALTY = -1e6


def num_words(word: str) -> float:
    frequency = word_frequency(word, "fr", wordlist="large", minimum=0.0)
    if frequency > 0:
        return True, math.log(frequency)
    return False, PENALTY


def clean(text):
    """
    :param text: question or answer text
    :return: text with line breaks and trailing spaces removed
    """
    return " ".join(text.strip().split())


def print_grid(letter_grid):
    for row in letter_grid:
        row = [" " if val == "" else val for val in row]
        print("".join(row), flush=True)


"""Pour calculer les performances des grilles de mots croiser apres scrapping"""


def performance(all_test_grid_in_our_format_grid):
    number_grid = 0
    number_correct_grid = 0
    list_letters_prop = []
    list_words_prop = []
    list_time = []
    for grid_obj in all_test_grid_in_our_format_grid:
        crossword = Crossword(grid_to_crossword(grid_obj))
        start = time.perf_counter()
        solver = BPSolver(crossword)
        solution = solver.solve()
        end = time.perf_counter()
        list_time.append(end - start)
        letters_correct_prop, words_correc_prop, grid_correct = solver.evaluate(
            solution
        )
        list_letters_prop.append(letters_correct_prop)
        list_words_prop.append(words_correc_prop)
        number_correct_grid += grid_correct
        number_grid += 1
    return (
        sum(list_letters_prop) / number_grid,
        sum(list_words_prop) / number_grid,
        number_correct_grid / number_grid,
        sum(list_time) / number_grid,
    )


# letter smoothing reprensente une liste de proba tel qu'un mot de tail n ne soit pas dans le answer set
def compute_letter_smoothing(set1, answer_set):
    matches = [0] * (23)

    counts_by_length = defaultdict(int)
    for w in set1:
        counts_by_length[len(w)] += 1

    for w in set1:
        l = len(w)
        if not (w in answer_set):
            matches[l] += 1

    # Calcul des probabilités
    LETTER_SMOOTHING_FACTOR = []
    for l in range(23):
        if counts_by_length[l] > 0:
            prob = matches[l] / counts_by_length[l]
        else:
            prob = 0.0
        LETTER_SMOOTHING_FACTOR.append(prob)

    print(LETTER_SMOOTHING_FACTOR)
