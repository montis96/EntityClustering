import sys

sys.path.append('.')
import Packages.ClusteringHelper as ch
from pyxdameraulevenshtein import damerau_levenshtein_distance
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import cdist


def main():
    text, data = ch.read_aida_yago_conll(
        "./aida-yago2-dataset/AIDA-YAGO2-dataset.tsv")

    ents_data = data[data['entities'] != '']
    mentions = ents_data['mentions'].values
    mentions = [x.lower() for x in mentions]

    def distance(x, y):
        x, y = x[0], y[0]
        if len(x) < 4 or len(y) < 4:
            if x == y:
                return 0
            else:
                return damerau_levenshtein_distance(x, y) + 3
        else:
            return damerau_levenshtein_distance(x, y)

    mentions_reshaped = np.array(mentions).reshape(-1, 1)
    print("Inizio il pairwise")
    m_matrix = cdist(mentions_reshaped, mentions_reshaped, metric=distance)
    print("Finito il pairwise")

    clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=1,
                                             linkage="single")
    cluster_numbers = clusterizator1.fit_predict(m_matrix)
    np.savetxt('damerau_1.txt', cluster_numbers, delimiter=',')


if __name__ == "__main__":
    main()
