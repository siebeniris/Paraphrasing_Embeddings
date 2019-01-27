import numpy as np
import torch
import torch.nn as nn

from models.plot_loss import plot_loss
from models.build_data import data_generator, random_train_dev_split
from models.build_data_ import data_generator_, random_train_dev_split_


def train_model(model, dir, optimizer,num_epochs,
                    entpair2id, path2id, batch_size, dev_ratio,  name, outputdir="", gpu =True):
    """ for SeqModel and UniModel"""

    if gpu:
        model.cuda()

    # check if model is on cuda
    print(next(model.parameters()).is_cuda)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer

    pos_neg_list= list(data_generator(dir, entpair2id,path2id))
    data = np.asarray(pos_neg_list)
    train_loader, dev_loader = random_train_dev_split(data, dev_ratio=dev_ratio, batch_size=batch_size)
    print("number of samples: ", len(pos_neg_list))

    cuda0 = torch.device('cuda:0')
    num_epochs = num_epochs

    loss_dict = {"train":[], "dev":[]}
    for epoch_nr in range(num_epochs):
        loss_accum = 0.0
        print("epoch", epoch_nr)
        for line in train_loader:
            for path_pos_neg in line:
                optimizer.zero_grad()
                diff = model.forward(path_pos_neg)
                label = torch.ones(diff.size(), dtype=torch.float32, device= cuda0) if gpu else\
                    torch.ones(diff.size(), dtype = torch.float32)
                loss = criterion(diff, label)
                loss_accum +=loss
                loss_dict["train"].append(loss_accum.detach().item()) # save gpu memory!
                loss.backward()
                optimizer.step()

                dev_loss= 0.0
                for line in dev_loader:
                    for path_pos_neg in line:
                        diff = model.forward(path_pos_neg)
                        label = torch.ones(diff.size(), dtype=torch.float32, device=cuda0) if gpu else \
                            torch.ones(diff.size(), dtype=torch.float32)
                        loss = criterion(diff, label)
                        dev_loss += loss
                        loss_dict["dev"].append(dev_loss.detach().item())
                print("dev loss:", dev_loss)
        print("current loss:", loss_accum)

    torch.save(model.state_dict(),outputdir+name+"_model.pt")
    plot_loss(loss_dict, outputdir + name + "_loss.png")


def train_word2vec(model,dir, optimizer, num_epochs,
                   entpair2id, path2id, batch_size,dev_ratio, name,  outputdir="",gpu=True):
    """ for Word2vec Model"""
    if gpu:
        model.cuda()
    # check if model i s on cuda

    print(next(model.parameters()).is_cuda)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optimizer

    pos_neg_list = list(data_generator(dir, entpair2id, path2id))
    data = np.asarray(pos_neg_list)
    train_loader, dev_loader = random_train_dev_split(data,dev_ratio=dev_ratio, batch_size=batch_size, gpu=gpu)
    print("samples: ", len(pos_neg_list))

    cuda0 = torch.device('cuda:0')
    num_epochs = num_epochs

    loss_dict = {"train":[], "dev":[]}
    for epoch_nr in range(num_epochs):
        loss_accum = 0.0
        print("epoch", epoch_nr)
        for line in train_loader:
            for path_pos_neg in line:
                optimizer.zero_grad()
                pos, neg = model.forward(path_pos_neg)
                pos_label = torch.ones(pos.size(), dtype=torch.float32, device=cuda0) if gpu else\
                            torch.ones(pos.size(), dtype=torch.float32)

                pos_loss = criterion(pos, pos_label)
                neg_label  = torch.zeros(neg.size(), dtype=torch.float32, device= cuda0) if gpu else \
                            torch.zeros(neg.size(), dtype=torch.float32)
                neg_loss = criterion(neg, neg_label)
                loss = pos_loss + neg_loss
                loss_accum += loss

                loss_dict["train"].append(loss_accum.detach().item())
                loss.backward()

                optimizer.step()

                dev_loss= 0.0
                for line in dev_loader:
                    for path_pos_neg in line:
                        pos, neg = model.forward(path_pos_neg)
                        pos_label = torch.ones(pos.size(), dtype=torch.float32, device= cuda0) if gpu else \
                                    torch.ones(pos.size(), dtype=torch.float32)
                        pos_loss = criterion(pos, pos_label)

                        neg_label = torch.zeros(neg.size(), dtype=torch.float32, device=cuda0) if gpu else \
                                    torch.zeros(neg.size(), dtype=torch.float32)
                        neg_loss = criterion(neg, neg_label)

                        loss = pos_loss + neg_loss
                        dev_loss += loss
                        loss_dict["dev"].append(dev_loss.detach().item())

                print("dev loss:", dev_loss)
        print("current loss:", loss_accum)

    torch.save(model.state_dict(), outputdir + name+"_model.pt")

    plot_loss(loss_dict, outputdir+name+"_loss.png")


def train_word2vec_ent(model,dir, optimizer, num_epochs,
                   entpair2id, path2id, ent2id, batch_size,dev_ratio, name,  outputdir="",gpu=True):
    """for word2vec model with both entity/entity_pair embeddings."""
    if gpu:
        model.cuda()

    # check if model i s on cuda
    print(next(model.parameters()).is_cuda)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optimizer

    pos_neg_list = list(data_generator_(dir, entpair2id, path2id, ent2id))
    data = np.asarray(pos_neg_list)
    train_loader, dev_loader = random_train_dev_split_(data,dev_ratio=dev_ratio, batch_size=batch_size, gpu=gpu)
    print("samples: ", len(pos_neg_list))

    cuda0 = torch.device('cuda:0')
    num_epochs = num_epochs

    loss_dict = {"train":[], "dev":[]}
    for epoch_nr in range(num_epochs):
        loss_accum = 0.0
        print("epoch", epoch_nr)
        for line in train_loader:
            for path_pos_neg in line:
                optimizer.zero_grad()
                pos, neg = model.forward(path_pos_neg)
                pos_label = torch.ones(pos.size(), dtype=torch.float32, device=cuda0) if gpu else\
                            torch.ones(pos.size(), dtype=torch.float32)
                pos_loss = criterion(pos, pos_label)

                neg_label  = torch.zeros(neg.size(), dtype=torch.float32, device= cuda0) if gpu else \
                            torch.zeros(neg.size(), dtype=torch.float32)
                neg_loss = criterion(neg, neg_label)

                loss = pos_loss + neg_loss
                loss_accum += loss

                loss_dict["train"].append(loss_accum.detach().item())
                loss.backward()

                optimizer.step()

                dev_loss= 0.0
                for line in dev_loader:
                    for path_pos_neg in line:
                        pos, neg = model.forward(path_pos_neg)
                        pos_label = torch.ones(pos.size(), dtype=torch.float32, device= cuda0) if gpu else \
                                    torch.ones(pos.size(), dtype=torch.float32)
                        pos_loss = criterion(pos, pos_label)

                        neg_label = torch.zeros(neg.size(), dtype=torch.float32, device=cuda0) if gpu else \
                                    torch.zeros(neg.size(), dtype=torch.float32)
                        neg_loss = criterion(neg, neg_label)

                        loss = pos_loss + neg_loss
                        dev_loss += loss
                        loss_dict["dev"].append(dev_loss.detach().item())

                print("dev loss:", dev_loss)
        print("current loss:", loss_accum)

    torch.save(model.state_dict(), outputdir + name+"_model.pt")

    plot_loss(loss_dict, outputdir+name+"_loss.png")


def train_model_ent(model, dir, optimizer,num_epochs,
                    entpair2id, path2id, ent2id, batch_size, dev_ratio,  name, outputdir="", gpu =True):
    """for seqmodel and unimodel using both entity/entity_pair embeddings"""

    if gpu:
        model.cuda()

    #check if model is on cuda
    print(next(model.parameters()).is_cuda)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer
    pos_neg_list= list(data_generator_(dir, entpair2id, path2id, ent2id))
    data = np.asarray(pos_neg_list)
    train_loader, dev_loader = random_train_dev_split_(data, dev_ratio=dev_ratio, batch_size=batch_size)
    print("samples: ", len(pos_neg_list))

    cuda0 = torch.device('cuda:0')
    num_epochs = num_epochs

    loss_dict = {"train":[], "dev":[]}
    for epoch_nr in range(num_epochs):
        loss_accum = 0.0
        print("epoch", epoch_nr)
        for line in train_loader:
            for path_pos_neg in line:
                optimizer.zero_grad()
                diff = model.forward(path_pos_neg)
                label = torch.ones(diff.size(), dtype=torch.float32, device= cuda0) if gpu else\
                    torch.ones(diff.size(), dtype = torch.float32)
                loss = criterion(diff, label)
                loss_accum +=loss
                loss_dict["train"].append(loss_accum.detach().item()) # save gpu memory!
                loss.backward()
                optimizer.step()

                dev_loss= 0.0
                for line in dev_loader:
                    for path_pos_neg in line:
                        diff = model.forward(path_pos_neg)
                        label = torch.ones(diff.size(), dtype=torch.float32, device=cuda0) if gpu else \
                            torch.ones(diff.size(), dtype=torch.float32)
                        loss = criterion(diff, label)
                        dev_loss += loss
                        loss_dict["dev"].append(dev_loss.detach().item())
                print("dev loss:", dev_loss)
        print("current loss:", loss_accum)

    torch.save(model.state_dict(),outputdir+name+"_model.pt")
    plot_loss(loss_dict, outputdir + name + "_loss.png")
