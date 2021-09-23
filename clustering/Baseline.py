import sys

sys.path.append('.')
import Packages.ClusteringHelper as ch
from textdistance import JaroWinkler
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def main():
    text, data = ch.read_aida_yago_conll(
        "D:\\Sgmon\\Documents\\Magistrale\\TESI\\ClusteringAndLinking\\aida-yago2-dataset\\AIDA-YAGO2-dataset.tsv")

    data = ch.filter_data(data, 3)
    n_entities = sum([x is not '' for x in list(data['entities'])])
    n_ass_ents = sum([x is not '' for x in list(data['numeric_codes'])])
    # n_tokens = sum([1 for x in list(data['entities'])])
    n_tokens = sum([len(x.split()) for x in text])
    golden_standard_dict = ch.get_gold_standard_dict(data)

    ents_data = data[data['entities'] != '']
    # golden_standard_entities = ents_data['entities'].values
    mentions = ents_data['mentions'].values
    mentions = [x.lower() for x in mentions]

    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        if len(mentions[i]) < 4:
            if mentions[i] == mentions[j]:
                return 0
            else:
                return JaroWinkler().distance(mentions[i].lower(), mentions[j].lower()) + 3
        else:
            return JaroWinkler().distance(mentions[i].lower(), mentions[j].lower())

    X = np.arange(len(mentions)).reshape(-1, 1)
    print("Inizio il pairwise")
    m_matrix = pairwise_distances(X, X, metric=lev_metric, n_jobs=-1)
    # clusterizator1 = DBSCAN(metric=dam_lev_metric, eps=1, min_samples=0, n_jobs=-1)
    print("Finito il pairwise")

    clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=0.2,
                                             linkage="single")
    cluster_numbers = clusterizator1.fit_predict(m_matrix)
    np.savetxt('db_cluster_dam_agglom.txt', cluster_numbers, delimiter=',')


if __name__ == "__main__":
    main()
