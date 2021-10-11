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
from pyxdameraulevenshtein import damerau_levenshtein_distance


def main(argv):
    original_stdout = sys.stdout
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    opts, _ = getopt.getopt(argv, "s:f:e:r:d:t",
                            ["step=", "first_threshold=", "second_threshold=", "randomly", "seed=", "entropy="])
    step = 10
    first_threshold = 0.035
    second_threshold = 0.015
    entropy = 25
    seed = None
    randomly = False
    os.makedirs("./Results/" + now)
    for opt, arg in opts:
        if opt in ("-s", "--step"):
            step = int(arg)
        elif opt in ("-ft", "--first_threshold"):
            first_threshold = float(arg)
        elif opt in ("-st", "--second_threshold"):
            second_threshold = float(arg)
        elif opt in ("-sd", "--seed"):
            seed = int(arg)
        elif opt in ("-t", "--entropy"):
            entropy = int(arg)
        elif opt in ("-r", "--randomly"):
            randomly = True

    with open("./Results/" + now + "/settings.txt", "a") as f:
        sys.stdout = f
        print('step:', step)
        print('first_threshold:', first_threshold)
        print('second_threshold:', second_threshold)
        print('seed:', seed)
        print('randomly:', randomly)
        print('Mean')
        print('Full_HAC')
        print('DamerauLevenshtein = 1')
        print('Threshold broke cluster: ', entropy)
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
    gold_entities = []
    total_clusters = []
    n = 0

    ## Let the cycle start

    tic = time.perf_counter()
    times = []
    for iteration in tqdm(evolving, total=math.ceil(len(evolving.documents) / evolving.step)):
        current_mentions = list(evolving.get_current_data().mentions)
        current_encodings = list(evolving.get_current_data()['encodings'])
        current_entities = list(evolving.get_current_data()['entities'])

        def dam_lev_metric(x, y):
            i, j = x[0], y[0]
            if len(i) < 4 or len(j) < 4:
                if i == j:
                    return 0
                else:
                    return damerau_levenshtein_distance(i.lower(), j.lower()) + 3
            else:
                return damerau_levenshtein_distance(i.lower(), j.lower())

        if len(current_mentions) == 1:
            pass
        if len(current_mentions) == 1:
            cluster_numbers = np.zeros(1, dtype=np.int8)
        else:
            X = np.array(current_mentions).reshape(-1, 1)
            m_matrix = cdist(X, X, metric=dam_lev_metric)
            # clusterizator1 = DBSCAN(metric=dam_lev_metric, eps=1, min_samples=0, n_jobs=-1)
            clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                     distance_threshold=1.2,
                                                     linkage="single")
            cluster_numbers = clusterizator1.fit_predict(m_matrix)

        cee_dict = {k: {'entities': [], 'mentions': [], 'encodings': [], 'sotto_clusters': None} for k in
                    set(cluster_numbers)}
        for i, cluster in enumerate(cluster_numbers):
            cee_dict[cluster]['entities'].append(current_entities[i])
            cee_dict[cluster]['mentions'].append(current_mentions[i])
            cee_dict[cluster]['encodings'].append(current_encodings[i])
        cee_list = cee_dict.values()
        clusterizator2 = AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=first_threshold,
                                                 linkage="single")
        for cluster in cee_dict.keys():
            try:
                cee_dict[cluster]['sotto_clusters'] = clusterizator2.fit_predict(cee_dict[cluster]['encodings'])
            except ValueError:
                cee_dict[cluster]['sotto_clusters'] = np.zeros(1, dtype=np.int8)

        sottocluster_list = []
        for el in cee_list:
            sotto_cluster = {k: Cluster() for k in set(el['sotto_clusters'])}
            for i, key in enumerate(el['sotto_clusters']):
                sotto_cluster[key].add_element(mention=el['mentions'][i], entity=el['entities'][i],
                                               encodings=el['encodings'][i])
            sottocluster_list.append(sotto_cluster)
        sottocluster_list = [clusters_dict[key] for clusters_dict in sottocluster_list for key in clusters_dict]

        current_clusters = total_clusters + sottocluster_list
        sotto_encodings = [x.encodings_mean() for x in current_clusters]
        if len(sotto_encodings) == 1:
            cluster_numbers = np.zeros(1, dtype=np.int8)
        else:
            clusterizator3 = AgglomerativeClustering(n_clusters=None, affinity='cosine',
                                                     distance_threshold=second_threshold,
                                                     linkage="single")
            cluster_numbers = clusterizator3.fit_predict(sotto_encodings)
        final_clusters = {k: Cluster() for k in set(cluster_numbers)}

        last_key = list(set(final_clusters.keys()))[-1]

        for i, x in enumerate(current_clusters):
            if compare_ecoding(final_clusters[cluster_numbers[i]], x):
                final_clusters[cluster_numbers[i]] = final_clusters[cluster_numbers[i]] + x
            else:
                print('dentro')
                last_key = last_key + 1
                final_clusters[last_key] = x

        gold_entities = gold_entities + current_entities
        total_clusters = list(final_clusters.values())

        broken_cluster = []
        to_remove_cluster = []
        for cl_index, cl in enumerate(total_clusters):
            if len(set([men.lower() for men in cl.mentions])) > entropy:
                X = np.array(cl.mentions).reshape(-1, 1)
                m_sub_matrix = cdist(X, X, metric=dam_lev_metric)
                br_clusterizator = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                           distance_threshold=1.2,
                                                           linkage="single")
                br_cluster_number = br_clusterizator.fit_predict(m_sub_matrix)

                br_cluster_dict = {k: Cluster() for k in set(br_cluster_number)}
                for i, cluster in enumerate(br_cluster_number):
                    br_cluster_dict[cluster].add_element(cl.mentions[i], cl.entities[i], cl.encodings_list[i])

                broken_cluster = broken_cluster + list(br_cluster_dict.values())
                to_remove_cluster.append(cl_index)
        for i in sorted(to_remove_cluster, reverse=True):
            del total_clusters[i]
        total_clusters = total_clusters + broken_cluster

        # total_clusters = [x for x in total_clusters if len(set([men.lower() for men in x.mentions])) < 15]

        # BCUBED
        bcubed_precision, bcubed_recall = ch.calcolo_b_cubed(total_clusters, gold_entities)
        bcubed_f1 = (2 * (bcubed_recall * bcubed_precision)) / (bcubed_precision + bcubed_recall)

        # CEAFm
        best_alignment = ch.get_optimal_alignment([x.count_ents() for x in total_clusters], set(gold_entities),
                                                  is_dict=False)
        CEAFm_p = sum(best_alignment.values()) / len(gold_entities)
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
            print(dict(Counter(gold_entities)))
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
