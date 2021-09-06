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
        self.count_ments = Counter(self.mentions)
        self.count_ents = Counter(self.entities)
        self.encodings_mean = np.mean(self.encodings_list, axis=0) if len(self.encodings_list) > 0 else np.zeros(1024)
        self.n_elements = len(self.mentions)

    def add_element(self, mention, entity, encodings):
        self.mentions.append(mention)
        self.entities.append(entity)
        self.encodings_list.append(encodings)
        self.count_ments = Counter(self.mentions)
        self.count_ents = Counter(self.entities)
        self.encodings_mean = np.mean(self.encodings_list, axis=0)
        self.n_elements = len(self.mentions)

    def __add__(self, other):
        return Cluster(mentions=self.mentions + other.mentions, entities=self.entities + other.entities,
                       encodings_list=self.encodings_list + other.encodings_list)

    def __repr__(self):
        return "Cluster" + self.count_ents.__repr__()

    def __str__(self):
        return "Cluster" + self.count_ents.__str__()
