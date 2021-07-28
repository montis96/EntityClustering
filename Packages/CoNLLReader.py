import pandas as pd
import re
import codecs
from pathlib import Path
from nltk import WhitespaceTokenizer

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
                documents.append(document)

                try:
                    if word[1] == "B":
                        tokens.append(word[2])
                except:
                    tokens.append(word[0])
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
            document = int(word[0].split()[1][1:].replace("testa", "").replace("testb", ""))
    for i in range(len(tags)):
        if entities[i] == '--NME--':
            entities[i] = ''
            tags[i] = ''
            mentions[i] = ''
    dataframe = pd.DataFrame(
        list(zip(documents, tokens, indexes, tags, mentions, entities, wikidatas, numeric_codes, alpha_codes)),
        columns=["documents", "tokens", "indexes", "tags", "mentions", "entities", "wikidatas",
                 "numeric_codes", "alpha_codes"])
    # text = text_reconstruction(dataframe)
    return 'text', dataframe


# def text_reconstruction(dataframe):


