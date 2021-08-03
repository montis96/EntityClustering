#%%

import Packages.ClusteringHelper as ch
from tqdm import tqdm
from Levenshtein import distance
from collections import Counter

#%%

text, data = ch.read_aida_yago_conll(
    "D:\\Sgmon\\Documents\\Magistrale\\TESI\\ClusteringAndLinking\\aida-yago2-dataset\\AIDA-YAGO2-dataset.tsv")
save = False
if save:
    text_file = open('text.txt', 'w')
    text_file.write(text)
    text_file.close()
data
#%%

data = ch.filter_data(data, 3)

#%% md

### General Info

#%%

n_entities = sum([x is not '' for x in list(data['entities'])])
n_ass_ents = sum([x is not '' for x in list(data['numeric_codes'])])
# n_tokens = sum([1 for x in list(data['entities'])])
n_tokens = sum([len(x.split()) for x in text])


#%%

print('{0:<35} {1:>10} '.format("Numero totale di entità:", n_entities))
print('{0:<35} {1:>10} '.format("Numero totale di tokens:", n_tokens))
print('{0:<35} {1:>10} '.format("1 entità ogni:", round(n_tokens / n_entities, 2)))

#%% md

### Gold standard

#%%

golden_standard_dict = ch.get_gold_standard_dict(data)
#%%
ents_data = data[data['entities'] != '']
golden_standard_entities = ents_data['entities'].values
mentions = ents_data['mentions'].values
mentions = [x.lower() for x in mentions]

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
    np.savetxt('db_cluster_levestein_0_3.txt', leven_cluster, delimiter=',')
else:
    leven_cluster = np.loadtxt("../aida-yago2-dataset/db_cluster_levestein_3_3.txt", dtype=np.int32)
