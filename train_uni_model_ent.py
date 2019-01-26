import json

import torch.optim as optim

from models.train_loop import train_model_ent
from models.uni_model_ent import UniModel

dir = "data_/"

entpair2id = json.load(fp=open(dir + "entpair2id.json"))
path2id = json.load(fp=open(dir + "path2id.json"))
ent2id = json.load(fp= open(dir+"ent2id.json"))

exp_dir = "experiments/"

uni_model_dir = exp_dir+"unimodel_ent/"

log = open("experiments/log_unimodel_ent.txt","a")

model = UniModel(entpair_size=len(entpair2id), path_size=len(path2id), ent_size=len(ent2id), embed_size=300)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epoch = 20
batch_size = 4096
log.write(str(num_epoch) + "\n")

train_model_ent(model, dir=dir, optimizer=optimizer, num_epochs=num_epoch,
                        entpair2id=entpair2id, path2id=path2id, ent2id=ent2id, batch_size=batch_size, dev_ratio=0.01,
                        name='adam_0.01'+"_"+str(num_epoch)+"_"+str(batch_size), outputdir=uni_model_dir, gpu=True)
log.write("+++++++++++++++++++++++++\n")

