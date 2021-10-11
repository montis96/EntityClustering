import getopt
import sys

sys.path.append('.')

import Packages.ClusteringHelper as ch
from Packages.TimeEvolving import DataEvolver, compare_ecoding
# from textdistance import DamerauLevenshtein, Levenshtein, JaroWinkler
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from Packages.TimeEvolving import Cluster
from tqdm import tqdm
import math
from collections import Counter
import datetime, time, os
from scipy.spatial.distance import cdist
from pyxdameraulevenshtein import damerau_levenshtein_distance, damerau_levenshtein_distance_seqs


def main(argv):
    original_stdout = sys.stdout
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    opts, _ = getopt.getopt(argv, "s:f:e:r:d:t",
                            ["step=", "first_threshold=", "second_threshold=", "randomly", "seed=", "entropy="])
    step = 10
    seed = 42
    randomly = True
    os.makedirs("./Results/" + now)
    for opt, arg in opts:
        if opt in ("-s", "--step"):
            step = int(arg)
        elif opt in ("-sd", "--seed"):
            seed = int(arg)
        elif opt in ("-r", "--randomly"):
            randomly = True

    with open("./Results/" + now + "/settings.txt", "a") as f:
        sys.stdout = f
        print('step:', step)
        print('seed:', seed)
        print('randomly:', randomly)
        print('BASELINE')
        print('Full_HAC')
        print('DamerauLevenshtein = 1')
        # print('Threshold dot_product')
        sys.stdout = original_stdout
    text, data = ch.read_aida_yago_conll(
        "./aida-yago2-dataset/AIDA-YAGO2-dataset.tsv")
    save = False
    if save:
        text_file = open('text.txt', 'w')
        text_file.write(text)
        text_file.close()
    ents_data = data[data['entities'] != ''].copy()

    ents_data = ch.add_entities_embedding(ents_data,
                                          "./aida-yago2-dataset/encodings")
    # ents_data_filtered = ents_data.copy()
    documents = set(ents_data.documents)

    evolving = DataEvolver(documents, ents_data, randomly=randomly, step=step, seed=seed)
    n = 0

    ## Let the cycle start
    total_mentions = []
    total_entities = []
    tic = time.perf_counter()
    times = []
    for iteration in tqdm(evolving, total=math.ceil(len(evolving.documents) / evolving.step)):
        total_mentions = total_mentions + list(evolving.get_current_data().mentions)
        total_mention_lower = [x.lower() for x in total_mentions]
        total_entities = total_entities + list(evolving.get_current_data().entities)
        mention_counter = {k: [] for k in set(total_mention_lower)}
        for index, men in enumerate(total_mention_lower):
            mention_counter[men].append(total_entities[index])

        def dam_lev_metric(x, y):
            i, j = x[0], y[0]
            if len(i) < 4 or len(j) < 4:
                if i == j:
                    return 0
                else:
                    return damerau_levenshtein_distance(i.lower(), j.lower()) + 3
            else:
                return damerau_levenshtein_distance(i.lower(), j.lower())

        if len(total_mentions) == 1:
            cluster_numbers = np.zeros(1, dtype=np.int8)
        else:
            X = np.array(list(set(total_mention_lower))).reshape(-1, 1)
            m_matrix = cdist(X, X, metric=dam_lev_metric)
            # clusterizator1 = DBSCAN(metric=dam_lev_metric, eps=1, min_samples=0, n_jobs=-1)
            clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                     distance_threshold=1.2,
                                                     linkage="single")
            cluster_numbers = clusterizator1.fit_predict(m_matrix)
        mention_to_cluster = {}
        for index, x in enumerate(set(total_mention_lower)):
            mention_to_cluster[x] = cluster_numbers[index]

        total_clusters = {k: Cluster() for k in set(cluster_numbers)}
        for index in range(len(total_mention_lower)):
            total_clusters[mention_to_cluster[total_mention_lower[index]]].add_element(
                total_mentions[index], total_entities[index], []
            )
        total_clusters = list(total_clusters.values())

        # BCUBED
        bcubed_precision, bcubed_recall = ch.calcolo_b_cubed(total_clusters, total_entities)
        bcubed_f1 = (2 * (bcubed_recall * bcubed_precision)) / (bcubed_precision + bcubed_recall)
        # CEAFm
        best_alignment = ch.get_optimal_alignment([x.count_ents() for x in total_clusters], set(total_entities),
                                                  is_dict=False)
        CEAFm_p = sum(best_alignment.values()) / len(total_entities)
        CEAFm_r = sum(best_alignment.values()) / sum([x.n_elements() for x in total_clusters])
        CEAFm_f1 = 2 * (CEAFm_p * CEAFm_r) / (CEAFm_p + CEAFm_r)
        with open("./Results/" + now + "/step" + str(n) + ".html", "a", encoding='utf-8') as f:
            sys.stdout = f
            print('<html>')
            print("Documents:", iteration, '<br>')
            print("CEAFm-R:", CEAFm_r)
            print("CEAFm-P:", CEAFm_p)
            print("CEAFm:", CEAFm_f1)
            print("<br>")
            print("bcubed_recall:", bcubed_recall)
            print("bcubed_precision:", bcubed_precision)
            print("bcubed_f1:", bcubed_f1)
            print("<br> Time:", time.perf_counter() - tic, '<br>')
            print("<br>", "Clusters:", '<br>')
            print(*total_clusters, sep=" <br><br> ")
            print("<br>")
            print("<br>")
            print("Gold_standard:", '<br>')
            print(dict(Counter(total_entities)))
            print('</html>')
            sys.stdout = original_stdout
        n = n + 1
        times.append(time.perf_counter() - tic)

    toc = time.perf_counter()
    with open("./Results/" + now + "/settings.txt", "a") as f:
        sys.stdout = f
        print('time:', toc - tic)
        print("Times:", times)
        sys.stdout = original_stdout
    print("Time:", toc - tic)


if __name__ == "__main__":
    main(sys.argv[1:])
