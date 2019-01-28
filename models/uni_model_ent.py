import torch
import torch.nn as nn


class UniModel(nn.Module):
    def __init__(self, entpair_size,  ent_size,path_size, embed_size):
        super(UniModel, self).__init__()
        self.embed_size = embed_size
        self.embeddings_entity = nn.Embedding(num_embeddings=ent_size, embedding_dim=embed_size)
        self.embeddings_entities = nn.Embedding(num_embeddings=entpair_size, embedding_dim=embed_size)
        self.embeddings_paths = nn.Embedding(num_embeddings=path_size, embedding_dim=embed_size)

        # initialize the word vectors so that the components are normally distributed
        # with mean 0 and variance 0.1
        self.embeddings_entities.weight.data.normal_(mean=0, std=0.1)
        self.embeddings_paths.weight.data.normal_(mean=0, std=0.1)
        self.embeddings_entity.weight.data.normal_(mean=0, std=0.1)

    def forward(self, ents_path_idxs):
        pos_idxs= ents_path_idxs[:,1]  # (batch_size,1)
        neg_idxs = ents_path_idxs[:,2]  # (batch_size, 1)

        # entities for pos_samples
        ent0, ent1 = ents_path_idxs[:, 3], ents_path_idxs[:, 4]
        # entities for neg_samples
        ent2, ent3 = ents_path_idxs[:, 5], ents_path_idxs[:, 6]

        # entity_pair embeddings + entity_embeddings
        pos_vecs = self.embeddings_entities(pos_idxs) + self.embeddings_entity(ent0) + self.embeddings_entity(ent1)
        neg_vecs = self.embeddings_entities(neg_idxs) + self.embeddings_entity(ent2) + self.embeddings_entity(ent3)

        pos_vecs = pos_vecs.view(-1, 1, self.embed_size)
        neg_vecs = neg_vecs.view(-1, 1,self.embed_size)

        path_idxs = ents_path_idxs[:,0]
        path_vecs = self.embeddings_paths(path_idxs)
        path_vecs = path_vecs.view(-1,self.embed_size,1)

        positive_predictions = torch.bmm(pos_vecs, path_vecs)
        negative_predictions = torch.bmm(neg_vecs, path_vecs)

        diff = positive_predictions-negative_predictions

        return diff

