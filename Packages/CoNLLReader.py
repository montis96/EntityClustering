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
    for raw_word in raw_line:
        word = raw_word.split("\t")
        if len(word[0]) > 0 and "DOCSTART" not in word[0]:
            tokens.append(word[0])
            try:
                tags.append(word[1])
            except:
                tags.append(None)
            try:
                mention.append(word[2])
            except:
                mention.append(None)
            try:
                entities.append(word[3])
            except:
                entities.append(None)
            try:
                wikidatas.append(word[4])
            except:
                wikidatas.append(None)
            try:
                numeric_codes.append(word[5])
            except:
                numeric_codes.append(None)
            try:
                alpha_codes.append(word[6])
            except:
                alpha_codes.append(None)
    text = " ".join(tokens)
    indexes = list(WhitespaceTokenizer().span_tokenize(text))
    return text, pd.DataFrame(list(zip(tokens, indexes, tags, mention, entities, wikidatas, numeric_codes, alpha_codes)),
                        columns=["tokens", "indexes", "tags", "mention", "entities", "wikidatas", "numeric_codes",
                                 "alpha_codes"])  # %%
