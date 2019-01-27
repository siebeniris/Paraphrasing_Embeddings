import json

import torch


def load_state_dict(model, path, path2id, output):
    model.load_state_dict(torch.load(path))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(param_tensor, "\t", model.state_dict()[param_tensor])


    word2vector = {}
    for key, i in path2id.items():
        vec = model.embeddings_paths.weight.data[i].numpy()
        vec = ' '.join([v for v in str(vec)[1:-1].split()])
        word2vector[key]=vec

    with open(output, "w") as f:
        f.write('{} {}\n'.format(len(word2vector), 300))
        for word, vec in word2vector.items():
            line = '{} {}\n'.format(word,vec)
            f.write(line)


if __name__ == '__main__':
    dir = "data_/"
    entpair2id = json.load(fp=open(dir + "entpair2id.json"))
    path2id = json.load(fp=open(dir + "path2id.json"))
    ent2id = json.load(fp=open(dir + "ent2id.json"))

    m = "w2v_ent"
    if m == "uni":
        from models.uni_model import UniModel
        model = UniModel(len(entpair2id), len(path2id), embed_size=300)
    elif m == "w2v":
        from models.word2vec import Word2Vec
        model = Word2Vec(len(entpair2id), len(path2id), embed_size=300)
    elif m=="w2v_ent":
        from models.word2vec_ent import Word2Vec
        model= Word2Vec(len(entpair2id), len(path2id), len(ent2id), embed_size=300)
    elif m=="uni_ent":
        from models.uni_model_ent import UniModel
        model=UniModel(len(entpair2id),  len(ent2id),len(path2id), embed_size=300)
    else:
        from models.seq_model import SeqModel
        model = SeqModel(len(entpair2id), len(path2id), embed_size=300)

    path ="experiments/word2vec_ent/adam_0.001_40_4096_model.pt"

    output = path+"_embeddings.csv"

    load_state_dict(model, path, path2id, output)
