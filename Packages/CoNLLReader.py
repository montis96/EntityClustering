import pandas as pd
import re
from pathlib import Path
from nltk import WhitespaceTokenizer

"""
Parsing del file Ayda-yago CONLL
"""


def read_aida_yago_conll(raw_path):
    path = Path(raw_path)
    raw_text = path.read_text(encoding='utf-8').strip()
    raw_line = re.split(r'\n', raw_text)
    tokens = []
    tags = []
    mention = []
    entities = []
    wikidatas = []
    numeric_codes = []
    alpha_codes = []
    counter = 0
    indexes = []
    for index, raw_word in enumerate(raw_line):
        word = raw_word.split("\t")
        if len(word[0]) > 0 and "DOCSTART" not in word[0]:
            # ho riunito i token che sono stati assegnati alla stessa menzione
            try:
                if word[1] == "B":
                    tokens.append(word[2])
            except:
                tokens.append(word[0])
            try:
                if word[1] == "B":
                    tags.append(word[1])
            except:
                tags.append('')
            try:
                if word[1] == "B":
                    mention.append(word[2])
            except:
                mention.append('')
            try:
                if word[1] == "B":
                    entities.append(word[3])
            except:
                entities.append('')
            try:
                if word[1] == "B":
                    wikidatas.append(word[4])
            except:
                wikidatas.append('')
            try:
                if word[1] == "B":
                    numeric_codes.append(word[5])
            except:
                numeric_codes.append('')
            try:
                if word[1] == "B":
                    alpha_codes.append(word[6])
            except:
                alpha_codes.append('')
            try:
                if word[1] == "B":
                    indexes.append((counter, counter + len(tokens[-1])))
            except:
                indexes.append((counter, counter + len(tokens[-1])))
            try:
                if word[1] == "B":
                    counter = counter + len(tokens[-1]) + 1
            except:
                counter = counter + len(tokens[-1]) + 1

    text = " ".join(tokens)
    dataframe = pd.DataFrame(list(zip(tokens, indexes, tags, mention, entities, wikidatas, numeric_codes, alpha_codes)),
                             columns=["tokens", "indexes", "tags", "mentions", "entities", "wikidatas", "numeric_codes",
                                      "alpha_codes"])
    return text, dataframe
