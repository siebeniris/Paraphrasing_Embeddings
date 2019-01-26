#### in this file, we build pos, neg samples corresponding to each path
#### change working directory

import random
from collections import defaultdict

path_ent_pos = defaultdict(list)
with open("entities_path_filtered.txt")as f:

    for line in f.readlines():
        ents, path = line.split()
        if path in path_ent_pos.keys():
            path_ent_pos[path].append(ents)
        else:
            path_ent_pos[path] = [ents]

ents_list=[line.rstrip() for line in open("ents.txt").readlines()]

import itertools

path_ent_new = defaultdict(list)
for path in path_ent_pos.keys():

    pos_samples = path_ent_pos[path]
    total_neg_samples = list(set(ents_list)-set(pos_samples))
    for pos_sample in pos_samples:
        neg_samples = random.sample(total_neg_samples, k=5)
        pos_neg_pairs = [x for x in zip( itertools.repeat(pos_sample), neg_samples)]
        print(pos_neg_pairs)
        path_ent_new[path].append(pos_neg_pairs)


path_pos_neg_file = open("path_pos_neg.txt","w")
for path in path_ent_new.keys():
    pos_neg_list = path_ent_new[path]
    for line in pos_neg_list:
        for rank_pairs in line:
            pos,neg= rank_pairs
            path_pos_neg_file.write("{} {} {}\n".format(path, pos,neg))

