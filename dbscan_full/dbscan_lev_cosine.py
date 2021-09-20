import sys
sys.path.append('.')
sys.path.append('..')
import Packages.ClusteringHelper as ch
from textdistance import DamerauLevenshtein
import numpy as np
from sklearn.cluster import DBSCAN
import time

def main():
    original_stdout = sys.stdout
    with open("settings.txt", "a") as f:
        sys.stdout = f
        print("cosine_distance * 0.35 + damerau * 0.65")
        print("DBSCAN(eps=0.25, min_samples=3, n_jobs=-1)")
        print(time.perf_counter())
        sys.stdout = original_stdout
    text, data = ch.read_aida_yago_conll(
        "/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv")
    save = False
    if save:
        text_file = open('text.txt', 'w')
        text_file.write(text)
        text_file.close()
    ents_data = data[data['entities'] != ''].copy()
    ents_data = ch.add_entities_embedding(ents_data,
                                          "/aida-yago2-dataset/encodings")
    current_mentions = list(ents_data.mentions.values)
    current_encodings = list(ents_data['encodings'].values)
    current_entities = list(ents_data['entities'].values)

    from scipy.spatial.distance import cosine

    def metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        edit_distance = DamerauLevenshtein().normalized_distance(current_mentions[i].lower(),
                                                                 current_mentions[j].lower())
        cosine_distance = cosine(current_encodings[i], current_encodings[j])
        return cosine_distance * .35 + edit_distance * .65

    X = np.arange(len(current_mentions)).reshape(-1, 1)
    clusterizator1 = DBSCAN(metric=metric, eps=0.25, min_samples=3, n_jobs=-1)
    cluster_numbers = clusterizator1.fit_predict(X)
    np.savetxt('dbscan_lev_cosine.txt', cluster_numbers, delimiter=',')
    with open("settings.txt", "a") as f:
        print(time.perf_counter())


if __name__ == "__main__":
    main()
