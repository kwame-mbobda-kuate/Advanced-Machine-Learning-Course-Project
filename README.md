Plan de travail provisoire :
11/11 - 09/12 : En parallèle : récupération et nettoyage des données et en parallèle ; mise à jour de la base de code ;
09/12 - 30/12 : Entraînement des modèles et expériences ;
30/12 - 16/01 : Rédaction du rapport.

Sources de données à considérer :

1) Pour les mots-croisés et mots-fléchés :
    - https://www.mots-croises.ch/, https://www.mots-croises.ch/dictionnaire.htm (Système indépendant)
    - https://jeux.lemonde.fr/game-list/mots-croises, https://jeux.lemonde.fr/game-list/mini-mots-croises (Système indépendant)
    - https://www.la-croix.com/mots-croises (RCI)
    - https://jeux.franceinfo.fr/mots-croises/, https://jeux.franceinfo.fr/mots-fleches/ (MyGamify)
    - https://www.leparisien.fr/jeux/mots-croises/, https://www.leparisien.fr/jeux/mots-fleches/ (MyGamify)
    - https://www.notretemps.com/jeux/jeux-en-ligne/mots-croises, https://www.notretemps.com/jeux/jeux-en-ligne/mots-fleches (RCI)
    - https://lebelage.ca/jeux/mots-croises-2/, https://lebelage.ca/jeux/mots-fleches/ (RCI)
    - https://jeux.lefigaro.fr/mots-fleches (Système indépendant)
    - https://www.lecanardenchaine.fr/mots-croises/ (Système indépendant)
    - https://www.lanouvellerepublique.fr/loisirs/jeux/mots-croises, https://www.lanouvellerepublique.fr/loisirs/jeux/mots-fleches (RCI)
    - https://www.tf1info.fr/jeux/mots-croises/, https://www.tf1info.fr/jeux/mots-fleches/ (RCI)
    - https://www.cnews.fr/jeux/mots-croises, https://www.cnews.fr/jeux/mots-fleches (RCI)
    - https://www.maxi-mag.fr/jeux/mots-croises, https://www.maxi-mag.fr/jeux/mots-fleches (RCI)
    - https://www.canadafrancais.com/mots-croises/ (Système indépendant)
    - https://www.crosswordgiant.com/fr (Système indépendant)
    
2) Dictionnaire, base de savoir et corpus : https://dumps.wikimedia.org/, http://www.lexique.org/?page_id=378, http://www.lexique.org/?page_id=250&lang=en, https://www.atilf.fr/ressources/tlfi/


# Crossword Puzzle Resolution via Monte Carlo Tree Search

## Datasets (Puzzles)
### Standard test set
data/puzzles/nyt.new.ra.txt
### Hard test set
data/puzzles/nyt.new.ra.hard.txt
### Validation set
data/puzzles/nyt.valid.txt
### Training set
nyt.shuffle.txt

## Datasets (Clues)
### Seen clues for test set （conmpressed）
data/clues_before_2020-09-26
### Seen clues for validation set （conmpressed）
data/clues_before_2020-06-18

## Codes
### Algorithm implementation
- cps/search.py: The main class of the method with several versions of search method
    - class MCTS: the basic implementation of MCTS algorithm for CP
    - class MCTS_NM: MCTS algorithm with neural matching for clue retrieval
    - class Astar: A* algorithm
    - class LDS: Limited Discrepancy Search algorithm 
- cps/candgen.py: candidate generation module
- cps/cdbretr.py: seen clue retrievel module
    - class ClueES: clue retrieval with textual matching
    - class RAM_CTR: clue retrieval with neural matching
- cps/kbretr.py: knowledge base retrievel module
- cps/dictretr_v2.py: dictionary retrievel module
- cps/fillblank.py: blank filling module

### Evaluating on standard test set
> run_standard.py

### Evaluating on hard test set
> run_hard.py

### Aggregating test results
> aggres.py

### Generating data for reward function learning
> generate_data.py

### Training reward function
> train_reward.py

## Instructions

### Installing requirements by create a conda environment
> conda env create -f crossword.yml

### Compiling core modules with Cython to accelerate
> cythonize -a -i cps/puz_utils1.pyx

### Installing and start Elasticsearch service
[Download Elasticsearch](https://www.elastic.co/cn/downloads/elasticsearch)

start ES service (e.g. Windows, go to the ES installation directory): 
> .\bin\elasticsearch-service.bat start

### Download StanfordNLP model
> python -c "import stanfordnlp; stanfordnlp.download('en')"

## Other data (dictionaries, models, etc.)
[Link](https://u.pcloud.link/publink/show?code=XZvu7RVZJbsfpViTsRhJ0bDNb647lz8mJp57)
