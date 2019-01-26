import json
import torch.optim as optim

from models.train_loop import train_model


from models.seq_model import SeqModel

dir = "data_/"

entpair2id = json.load(fp=open(dir + "entpair2id.json"))
path2id = json.load(fp=open(dir + "path2id.json"))

exp_dir = "experiments/"

seq_model_dir = exp_dir+"seqmodel/"



models= [ "uni"]


log = open("experiments/log_word2vec.txt","a")



model = SeqModel(ent_size=len(entpair2id), path_size=len(path2id), embed_size=300)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epoch = 20
batch_size =4096
log.write("epochs"+str(num_epoch) + "\n")
log.write("batch_size"+str(batch_size)+'\n')
log.write("optimizer: adam , lr:0.01\n")

train_model(model, dir=dir, optimizer=optimizer, num_epochs=num_epoch,
                        entpair2id=entpair2id, path2id=path2id, batch_size=batch_size, dev_ratio=0.01,
                        name='linear_relu_2_adam_0.01'+"_"+str(num_epoch)+"_"+str(batch_size), outputdir=seq_model_dir, gpu=True)
log.write("+++++++++++++++++++++++++\n")
