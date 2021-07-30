import pandas as pd
import re
import codecs
from pathlib import Path
from nltk import WhitespaceTokenizer
import numpy as np

"""
Parsing del file Ayda-yago CONLL
"""


def read_aida_yago_conll(raw_path):
    path = Path(raw_path)
    raw_text = codecs.open(path, "r", "unicode_escape").read()
    raw_line = re.split(r'\n', raw_text)
    tokens = []
    tags = []
    mentions = []
    entities = []
    wikidatas = []
    numeric_codes = []
    alpha_codes = []
    counter = 0
    indexes = []
    documents = []
    document = ""
    for index, raw_word in enumerate(raw_line):
        word = raw_word.split("\t")
        if "DOCSTART" not in word[0]:
            if len(word[0]) > 0:
                # ho riunito i token che sono stati assegnati alla stessa menzione
                try:
                    if word[1] == "B":
                        tokens.append(word[2])
                        documents.append(document)
                except:
                    tokens.append(word[0])
                    documents.append(document)
                try:
                    if word[1] == "B" and word[4]:
                        tags.append(word[1])
                except:
                    tags.append('')
                try:
                    if word[1] == "B":
                        mentions.append(word[2])
                except:
                    mentions.append('')
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
        else:
            document = int(word[0].split()[1][1:].replace("testa", "").replace("testb", "")) - 1
            counter = 0

    for i in range(len(tags)):
        if entities[i] == '--NME--':
            entities[i] = ''
            tags[i] = ''
            mentions[i] = ''
    dataframe = pd.DataFrame(
        list(zip(documents, tokens, indexes, tags, mentions, entities, wikidatas, numeric_codes, alpha_codes)),
        columns=["documents", "tokens", "indexes", "tags", "mentions", "entities", "wikidatas",
                 "numeric_codes", "alpha_codes"])
    texts = text_reconstruction(dataframe)
    return texts, dataframe


def text_reconstruction(dataframe):
    texts = []
    for doc in np.unique(dataframe['documents'].values):
        texts.append(" ".join(dataframe[dataframe['documents'] == doc]['tokens'].values))
    return texts


def filter_data(dataframe, min_el):
    new_df = dataframe.copy()
    entities = new_df['entities']
    not_allowed = []
    from collections import Counter
    ent_dict = Counter(entities)
    for key in ent_dict:
        if ent_dict[key] < min_el:
            not_allowed.append(key)
    new_entities = []
    for x in new_df['entities']:
        if x in not_allowed:
            new_entities.append('')
        else:
            new_entities.append(x)
    new_df['entities'] = new_entities
    return new_df

