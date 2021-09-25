import sys

sys.path.append('.')
import Packages.ClusteringHelper as ch
from textdistance import JaroWinkler
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances

def main():
    text, data = ch.read_aida_yago_conll(
        "./aida-yago2-dataset/AIDA-YAGO2-dataset.tsv")

    ents_data = data[data['entities'] != '']
    mentions = ents_data['mentions'].values
    mentions = [x.lower() for x in mentions]

    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        return JaroWinkler().distance(mentions[i].lower(), mentions[j].lower())

    X = np.arange(len(mentions)).reshape(-1, 1)
    print("Inizio il pairwise")
    m_matrix = pairwise_distances(X, X, metric=lev_metric, n_jobs=-1)
    # clusterizator1 = DBSCAN(metric=lev_metric, eps=0.2, min_samples=0, n_jobs=-1)
    print("Finito il pairwise")

    clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=1,
                                             linkage="single")
    cluster_numbers = clusterizator1.fit_predict(m_matrix)
    np.savetxt('damerau_1.txt', cluster_numbers, delimiter=',')


if __name__ == "__main__":
    main()
