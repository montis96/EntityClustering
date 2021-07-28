#%%

import Packages.CoNLLReader as reader
import numpy as np
from tqdm import tqdm
from Levenshtein import distance

#%%

text, data = reader.read_aida_yago_conll(
    "D:\\Sgmon\\Documents\\Magistrale\\TESI\\ClusteringAndLinking\\aida-yago2-dataset\\AIDA-YAGO2-dataset.tsv")
save = False
if save:
    text_file = open('text.txt', 'w')
    text_file.write(text)
    text_file.close()
#%%

n_entities = sum([x is not '' for x in list(data['entities'])])
n_ass_ents = sum([x is not '' for x in list(data['numeric_codes'])])
n_tokens = sum([1 for x in list(data['entities'])])

#%%

print('{0:<35} {1:>10} '.format("Numero totale di entità:", n_entities))
print('{0:<35} {1:>10} '.format("Numero totale di tokens:", n_tokens))
print('{0:<35} {1:>10} '.format("1 entità ogni:", round(n_tokens / n_entities, 2)))

#%%

ent_nums = {}
ents_data = data[data['entities'] != '']
entities = ents_data['entities'].values
unique_entities = np.unique(entities)
for uniqu_ent in tqdm(entities):
    ent_nums[uniqu_ent] = 0
for uniqu_ent in tqdm(entities):
    ent_nums[uniqu_ent] = ent_nums[uniqu_ent] + 1
ent_nums = dict(sorted(ent_nums.items(), key=lambda item: item[1], reverse=True))
#%%
ents_data = data[data['entities'] != '']
mentions = ents_data['mentions'].values

mention_to_entity = {}
for men, ent in set(list(zip(list(ents_data['mentions']), list(ents_data['entities'])))):
    mention_to_entity[men] = ent
#%% md
### Golden standard
#%%
golden_standard = [mention_to_entity[x] for x in mentions]
golden_standard_dict = dict.fromkeys(golden_standard, 0)
for ent in golden_standard:
    golden_standard_dict[ent] = golden_standard_dict[ent] + 1
#%% md

### Clustering by Levenshtein distance and DBSCAN
#%%

# Way with dbscan algorithm
import numpy as np
from sklearn.cluster import dbscan

clustering = True
if clustering:
    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        if len(mentions[i]) < 4:
            if mentions[i] == mentions[j]:
                return 0
            else:
                return distance(mentions[i].lower(), mentions[j].lower()) + 3
        else:
            return distance(mentions[i].lower(), mentions[j].lower())


    X = np.arange(len(mentions)).reshape(-1, 1)
    _, leven_cluster = dbscan(X, metric=lev_metric, eps=1, min_samples=0, n_jobs=-1)
    np.savetxt('db_cluster_levestein.txt', leven_cluster, delimiter=',')
else:
    leven_cluster = np.loadtxt("../aida-yago2-dataset/db_cluster_levestein_0.txt", dtype=np.int32)
#%%
# Now we create a dict for each cluster taht contains entities and entities count
lev_cluster_dict = {}
for i, x in enumerate(leven_cluster):
    try:
        lev_cluster_dict[x].append(mentions[i])
    except:
        lev_cluster_dict[x] = [mentions[i]]
# del lev_cluster_dict[-1]
for key in tqdm(lev_cluster_dict):
    cluster_list = [mention_to_entity[x] for x in lev_cluster_dict[key]]
    cluster_set = set(cluster_list)
    cluster_dict = dict.fromkeys(list(cluster_set), 0)
    for cluster in cluster_list:
        cluster_dict[cluster] = cluster_dict[cluster] + 1
    lev_cluster_dict[key] = cluster_dict
#%% md

### evaluation levestein
#%%
max_lev_cluster_list = []
for key in lev_cluster_dict.keys():
    max_key = max(lev_cluster_dict[key], key=lev_cluster_dict[key].get)
    max_lev_cluster_list.append((max_key, lev_cluster_dict[key][max_key]))
max_lev_cluster_list.sort(key=lambda x: x[1], reverse=True)
max_lev_cluster_dict = dict.fromkeys(entities, 0)
for key, val in max_lev_cluster_list:
    if max_lev_cluster_dict[key] == 0:
        max_lev_cluster_dict[key] = val
#%%
# CEAFm_levenshtein_precision
CEAFm_levenshtein_precision = sum([x for x in max_lev_cluster_dict.values()]) / ents_data.shape[0]
#%%
# CEAFm_levenshtein_recall
CEAFm_levenshtein_recall = sum([x for x in max_lev_cluster_dict.values()]) / sum(
    [y for x in lev_cluster_dict.values() for y in x.values()])
#%%
# CEAFm_levenshtein_f1
CEAFm_levenshtein_f1 = (2 * (CEAFm_levenshtein_recall * CEAFm_levenshtein_precision)) / (
        CEAFm_levenshtein_precision + CEAFm_levenshtein_recall)
#%%
# B-cubed - recall
bcubed_recall_num = 0
for gold_key in tqdm(golden_standard_dict.keys()):
    for lev_key in lev_cluster_dict.keys():
        try:
            bcubed_recall_num = bcubed_recall_num + (pow(lev_cluster_dict[lev_key][gold_key], 2) /
                                       golden_standard_dict[gold_key])
        except:
            pass
bcubed_recall = bcubed_recall_num/ents_data.shape[0]
#%%
# B-cubed - precision
bcubed_precision_num = 0
for gold_key in tqdm(golden_standard_dict.keys()):
    for lev_key in lev_cluster_dict.keys():
        try:
            bcubed_precision_num = bcubed_precision_num + (pow(lev_cluster_dict[lev_key][gold_key], 2) /
                                       sum([x for x in lev_cluster_dict[lev_key].values()]))
        except:
            pass
bcubed_precision = bcubed_precision_num/sum([y for x in lev_cluster_dict.values() for y in x.values()])
#%%
bcubed_f1 = (2 * (bcubed_recall * bcubed_precision)) / (
        bcubed_precision + bcubed_recall)
#%%



