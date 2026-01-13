import json
import puz
import wordsegment
import math
from wordsegment import load, segment, clean

load()

# dictionary = set([a.strip() for a in open('solver/words_alpha.txt','r').readlines()])

#inutile dans notre cas
def num_words(fill):
    """segment the text into multiple words and count how many words the text has in total"""
    if fill not in dictionary: #constant
        return False,-9999999999999
    return True,math.log(wordsegment.UNIGRAMS[fill])


def get_word_flips(fill, num_candidates=10):
    """
    We take as input a word/phrase that is probably mispelled, something like iluveyou. We then try flipping each one of the letters
    to all other letters. We then segment those texts into multiple words using num_words, e.g., iloveyou -> i love you. We return the candidates
    that segment into the fewest number of words.
    """
    results = []
    fill = clean(fill)
    word_mispelled=False
    for index, char in enumerate(fill):
        for new_letter in "abcdefghijklmnopqrstuvwxyz":
            new_fill = list(fill)
            new_fill[index] = new_letter
            new_fill = "".join(new_fill)
            words_in_dictionnary,prob = num_words(new_fill)
            if words_in_dictionnary :
                word_mispelled = words_in_dictionnary
                results.append((new_fill.upper(),prob))
    if not(word_mispelled) :
        return [fill.upper()]
    
    results.sort(key=lambda x: -x[1])
    return [word for word, _ in results[:num_candidates]]


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


#nom_foncions supprimer :
""""convert_puz
"""
