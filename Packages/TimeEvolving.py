class DataEvolver:
    def __init__(self, documents, data, step=3, randomly=False):
        import random

        self.step = step
        self.documents = set(documents)
        self.data = data.copy()
        self.random = random
        self.done_docs = set()
        self.docs_stream = list(documents)
        if randomly:
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
