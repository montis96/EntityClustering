import pandas as pd
import re
import codecs
from pathlib import Path
from collections import Counter
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
    word_counter = 0
    word_indexes = []
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
                        tokens.append(word[0])
                        documents.append(document)
                    elif word[1] == "I":
                        tokens[-1] = tokens[-1] + " " + word[0]
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
                        word_indexes.append(word_counter)
                except:
                    indexes.append((counter, counter + len(tokens[-1])))
                    word_indexes.append(word_counter)
                try:
                    if word[1] == "B":
                        counter = counter + len(tokens[-1]) + 1
                        word_counter = word_counter + 1
                    else:
                        word_counter = word_counter + 1
                except:

                    counter = counter + len(tokens[-1]) + 1
                    word_counter = word_counter + 1
        else:
            document = int(word[0].split()[1][1:].replace("testa", "").replace("testb", "")) - 1
            counter = 0
            word_counter = 0

    for i in range(len(tags)):
        if entities[i] == '--NME--':
            entities[i] = ''
            tags[i] = ''
            mentions[i] = ''
    dataframe = pd.DataFrame(
        list(zip(documents, tokens, indexes, word_indexes, mentions, entities, wikidatas, numeric_codes, alpha_codes)),
        columns=["documents", "tokens", "indexes", 'word_indexes', "mentions", "entities", "wikidatas",
                 "numeric_codes", "alpha_codes"])
    texts = text_reconstruction(dataframe)
    return texts, dataframe


def text_reconstruction(dataframe):
    texts = []
    for doc in np.unique(dataframe['documents'].values):
        texts.append(" ".join(dataframe[dataframe['documents'] == doc]['tokens'].values))
    return texts


def get_gold_standard_dict(dataframe):
    ents_data = dataframe[dataframe['entities'] != '']
    golden_standard_entities = ents_data['entities'].values
    golden_standard_dict = Counter(golden_standard_entities)
    golden_standard_dict = dict(sorted(golden_standard_dict.items(), key=lambda item: item[1], reverse=True))
    return golden_standard_dict


def filter_data(dataframe, min_el):
    new_df = dataframe.copy()
    entities = new_df['entities']
    mentions = new_df['mentions']
    not_allowed = []
    from collections import Counter
    ent_dict = Counter(entities)
    for key in ent_dict:
        if ent_dict[key] < min_el:
            not_allowed.append(key)
    new_entities = []
    new_mentions = []
    for x in range(len(entities)):
        if entities[x] in not_allowed:
            new_entities.append('')
            new_mentions.append('')
        else:
            new_entities.append(entities[x])
            new_mentions.append(mentions[x])
    new_df['entities'] = new_entities
    new_df['mentions'] = new_mentions
    return new_df


def get_context(splitted_sentence, word_index, windows=128):
    half_window = round(windows / 2, 0)
    starting_index = word_index - half_window
    ending_index = word_index + half_window
    doc_len = len(splitted_sentence) - 1
    if starting_index < 0:
        ending_index = ending_index - starting_index
        starting_index = 0
        if ending_index > doc_len:
            ending_index = doc_len
    if ending_index > doc_len:
        starting_index = starting_index - ending_index + doc_len
        ending_index = doc_len
        if starting_index < 0:
            starting_index = 0
    return int(starting_index), int(ending_index)


def calculate_context_vector(model, cluster, text_splitted, doc_window=20, window=100, doc_context_alpha=0.4,
                             word_context_aplha=0.6):
    arr_length = len(model['word'])
    contexts_vectorized = []
    for el in cluster:
        doc_context_mean = np.zeros(arr_length)
        actual_doc_window = doc_window if doc_window < len(text_splitted[el[2]]) else len(text_splitted[el[2]])

        null_counter = 0
        for i in range(actual_doc_window):
            try:
                doc_context_mean += model[text_splitted[el[2]][i]]
            except:
                null_counter += 1
        doc_context_mean = doc_context_mean / (actual_doc_window - null_counter)

        null_counter = 0
        word_context_mean = np.zeros(arr_length)
        for word in text_splitted[el[2]][el[4][0]:el[4][1]]:
            try:
                word_context_mean += model[word]
            except:
                null_counter += 1

        word_context_mean = word_context_mean / (window - null_counter)
        final_mean = word_context_mean * word_context_aplha + doc_context_mean * doc_context_alpha
        contexts_vectorized.append(final_mean)
    return contexts_vectorized
