import json

import torch.optim as optim

from models.train_loop import train_word2vec_ent
from models.word2vec_ent import Word2Vec


dir = "data_/"

entpair2id = json.load(fp=open(dir + "entpair2id.json"))
path2id = json.load(fp=open(dir + "path2id.json"))
ent2id = json.load(fp= open(dir+"ent2id.json"))

exp_dir = "experiments/"
word2vec_dir = exp_dir +"word2vec_ent/"

log = open("experiments/log_word2vec_ent.txt","a")

model = Word2Vec(entpair_size=len(entpair2id),ent_size=len(ent2id), path_size=len(path2id), embed_size=300)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epoch = 40
batch_size = 4096
log.write(str(num_epoch) + "\n")

train_word2vec_ent(model, dir=dir, optimizer=optimizer, num_epochs=num_epoch,
                        entpair2id=entpair2id, path2id=path2id, ent2id = ent2id,batch_size=batch_size, dev_ratio=0.01,
                        name='adam_0.01'+"_"+str(num_epoch)+"_"+str(batch_size), outputdir=word2vec_dir, gpu=True)
log.write("+++++++++++++++++++++++++\n")
