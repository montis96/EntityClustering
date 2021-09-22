import sys

sys.path.append('.')
import Packages.ClusteringHelper as ch
from Packages.TimeEvolving import DataEvolver
from textdistance import DamerauLevenshtein, Levenshtein
import numpy as np
from sklearn.cluster import  AgglomerativeClustering
from Packages.TimeEvolving import Cluster
from tqdm import tqdm
import math
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import sys
import datetime, time, os
from collections import Counter

def main():
    original_stdout = sys.stdout
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    os.makedirs(".\\Results_one_step\\" + now)
    text, data = ch.read_aida_yago_conll(
        ".\\aida-yago2-dataset\\AIDA-YAGO2-dataset.tsv")
    save = False
    if save:
        text_file = open('text.txt', 'w')
        text_file.write(text)
        text_file.close()
    ents_data = data[data['entities'] != ''].copy()
    ents_data = ch.add_entities_embedding(ents_data,
                                          ".\\aida-yago2-dataset\\encodings")
    documents = set(ents_data.documents)
    evolving = DataEvolver(documents, ents_data, randomly=True, step=10, seed=42)
    gold_entities = []
    total_clusters = []
    n = 0

    tic = time.perf_counter()
    for iteration in tqdm(evolving, total=math.ceil(len(evolving.documents) / evolving.step)):
        current_mentions = list(evolving.get_current_data().mentions)
        current_encodings = list(evolving.get_current_data()['encodings'].values)
        current_entities = list(evolving.get_current_data()['entities'].values)
        gold_entities = gold_entities + current_entities
        current_clusters = [Cluster(mentions=[current_mentions[x]], encodings_list=[current_encodings[x]],
                                    entities=[current_entities[x]]) for x in range(len(current_mentions))]

        total_clusters = total_clusters + current_clusters

        def distance_metric(x, y):
            i, j = int(x[0]), int(y[0])  # extract indices
            sintact_distance_list = [DamerauLevenshtein().normalized_distance(el1, el2)
                                     for el1 in total_clusters[i].unique_mentions for el2 in
                                     total_clusters[j].unique_mentions]
            sintact_distance = np.mean(sintact_distance_list)
            semantic_distance = distance.cosine(total_clusters[i].encodings_mean, total_clusters[j].encodings_mean)
            return 0.63 * sintact_distance + 0.37 * semantic_distance

        X = np.arange(len(total_clusters)).reshape(-1, 1)
        m_matrix = pairwise_distances(X, X, metric=distance_metric, n_jobs=-1)
        clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                 distance_threshold=0.25,
                                                 linkage="single", memory='.\\cache')
        cluster_numbers = clusterizator1.fit_predict(m_matrix)
        cluster_dict = {x: Cluster() for x in set(cluster_numbers)}
        for index, x in enumerate(cluster_numbers):
            cluster_dict[x] = cluster_dict[x] + total_clusters[index]
        total_clusters = list(cluster_dict.values())
        # CEAFm
        best_alignment = ch.get_optimal_alignment([x.count_ents for x in total_clusters], set(gold_entities),
                                                  is_dict=False)
        CEAFm_p = sum(best_alignment.values()) / len(gold_entities)
        CEAFm_r = sum(best_alignment.values()) / sum([x.n_elements for x in total_clusters])
        CEAFm_f1 = 2 * (CEAFm_p * CEAFm_r) / (CEAFm_p + CEAFm_r)
        with open(".\\Results_one_step\\" + now + "\\step" + str(n) + ".html", "a", encoding='utf-8') as f:
            sys.stdout = f
            print('<html>')
            print("Documents:", iteration, '<br>')
            print("CEAFm-R:", CEAFm_r, '<br>')
            print("CEAFm-P:", CEAFm_p, '<br>')
            print("CEAFm:", CEAFm_f1, '<br>')
            print("Clusters:", '<br>')
            print(*total_clusters, sep=" <br><br> ")
            print("<br>")
            print("<br>")
            print("Gold_standard:", '<br>')
            print(dict(Counter(gold_entities)))
            print('</html>')
            sys.stdout = original_stdout
        n = n + 1
    print(time.perf_counter() - tic)


if __name__ == "__main__":
    main()
