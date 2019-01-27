import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, entpair_size, path_size, ent_size, embed_size):
        super(Word2Vec, self).__init__()
        self.embed_size = embed_size
        self.embeddings_entities = nn.Embedding(num_embeddings=entpair_size, embedding_dim=embed_size)
        self.embeddings_entity = nn.Embedding(num_embeddings=ent_size, embedding_dim=embed_size)
        self.embeddings_paths = nn.Embedding(num_embeddings=path_size, embedding_dim=embed_size)

        # initialize embeddings, normally distributed.
        self.embeddings_entities.weight.data.normal_(mean=0, std=0.1)
        self.embeddings_paths.weight.data.normal_(mean=0, std=0.1)
        # self.lstm = nn.LSTM(input_size=)


    def forward(self, ents_path_idxs):
        pos_idxs= ents_path_idxs[:,1]  # (batch_size,1)
        neg_idxs = ents_path_idxs[:,2]  # (batch_size, 1)

        # entities for positive entity pairs
        ent0, ent1 = ents_path_idxs[:,3], ents_path_idxs[:,4]
        # entities for negative entity pairs
        ent2, ent3 = ents_path_idxs[:,5], ents_path_idxs[:,6]

        pos_vecs = self.embeddings_entities(pos_idxs)+self.embeddings_entity(ent0)+self.embeddings_entity(ent1)
        neg_vecs = self.embeddings_entities(neg_idxs)+self.embeddings_entity(ent2)+self.embeddings_entity(ent3)

        pos_vecs = pos_vecs.view(-1, 1, self.embed_size)
        neg_vecs = neg_vecs.view(-1,1,self.embed_size)

        path_idxs = ents_path_idxs[:,0]
        paths_vecs = self.embeddings_paths(path_idxs).view(-1,self.embed_size,1)

        pos_prediction = torch.bmm(pos_vecs, paths_vecs)
        neg_prediction = torch.bmm(neg_vecs, paths_vecs)

        pos_score, neg_score =pos_prediction.view(-1,1), neg_prediction.view(-1,1)
        return pos_score, neg_score
