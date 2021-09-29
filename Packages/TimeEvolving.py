from collections import Counter
import numpy as np


class DataEvolver:
    def __init__(self, documents, data, step=3, randomly=False, seed=None):
        import random

        self.step = step
        self.documents = set(documents)
        self.data = data.copy()
        self.random = random
        self.done_docs = set()
        self.docs_stream = list(documents)
        if randomly:
            if seed is not None:
                random.Random(seed).shuffle(self.docs_stream)
            else:
                random.shuffle(self.docs_stream)
        self.current_docs = []

    def __iter__(self):
        return DataEvolverIterator(self)

    def get_current_data(self):
        return self.data[self.data['documents'].isin(self.current_docs)]


class DataEvolverIterator:
    def __init__(self, timer):
        self._evolver = timer
        self._end = len(self._evolver.documents)

    def __next__(self):
        to_return = self._evolver.docs_stream[:self._evolver.step]
        self._evolver.docs_stream = self._evolver.docs_stream[self._evolver.step:]
        self._evolver.done_docs.update(to_return)
        if len(to_return) == 0:
            raise StopIteration
        self._evolver.current_docs = to_return
        return to_return


class Cluster:
    def __init__(self, mentions=None, entities=None, encodings_list=None):
        if encodings_list is None:
            encodings_list = []
        if entities is None:
            entities = []
        if mentions is None:
            mentions = []

        self.encodings_list = encodings_list
        self.mentions = mentions
        self.entities = entities

    def add_element(self, mention, entity, encodings):
        self.mentions.append(mention)
        self.entities.append(entity)
        self.encodings_list.append(encodings)

    def count_ents(self):
        return Counter(self.entities)

    def count_ments(self):
        return Counter(self.mentions)

    def encodings_mean(self):
        return np.mean(self.encodings_list, axis=0) if len(self.encodings_list) > 0 else np.array([])

    def encodings_median(self):
        return np.median(self.encodings_list, axis=0) if len(self.encodings_list) > 0 else np.array([])

    def n_elements(self):
        return len(self.mentions)

    def unique_mentions(self):
        return list(set([x.lower() for x in self.mentions]))

    def __add__(self, other):
        return Cluster(mentions=self.mentions + other.mentions, entities=self.entities + other.entities,
                       encodings_list=self.encodings_list + other.encodings_list)

    def __repr__(self):
        return "Cluster" + self.print_to_html().__repr__() + "; #_elements = " + str(len(self.mentions))

    def __str__(self):
        return "Cluster" + self.print_to_html().__str__() + '; <span style="background: pink;">#_elements</span> = ' + str(
            len(self.mentions))

    def print_to_html(self):
        to_print = {"<b>" + x + "</b>": [] for x in set(self.entities)}
        for index in range(len(self.entities)):
            to_print["<b>" + self.entities[index] + "</b>"].append(self.mentions[index])
        for key in to_print:
            to_print[key] = dict(Counter(to_print[key]))
            to_print[key]['<span style="background: yellow;">#</span>'] = sum(to_print[key].values())
        return to_print


def compare_ecoding(cluster1, cluster2):
    if cluster1.n_elements() == 0:
        return True
    if cluster2.n_elements() == 0:
        return True
    for enc1 in cluster1.encodings_list:
        for enc2 in cluster1.encodings_list:
             if np.dot(enc1, enc2) > 80:
                 return True
    return False