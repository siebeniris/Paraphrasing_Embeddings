import json

import torch.optim as optim

from models.train_loop import train_model
from models.uni_model import UniModel

dir = "data_/"

entpair2id = json.load(fp=open(dir + "entpair2id.json"))
path2id = json.load(fp=open(dir + "path2id.json"))

exp_dir = "experiments/"

uni_model_dir = exp_dir+"unimodel/"



log = open("experiments/log_unimodel.txt","a")



model = UniModel(ent_size=len(entpair2id), path_size=len(path2id), embed_size=300)
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epoch = 20
batch_size = 4096
log.write(str(num_epoch) + "\n")

train_model(model, dir=dir, optimizer=optimizer, num_epochs=num_epoch,
                        entpair2id=entpair2id, path2id=path2id, batch_size=batch_size, dev_ratio=0.01,
                        name='sgd_0.01'+"_"+str(num_epoch)+"_"+str(batch_size), outputdir=uni_model_dir, gpu=True)
log.write("+++++++++++++++++++++++++\n")

