import torch
import torch.nn as nn


class SeqModel(nn.Module):
    def __init__(self, ent_size, path_size, embed_size):
        super(SeqModel, self).__init__()
        self.embed_size = embed_size
        self.embeddings_entities = nn.Embedding(num_embeddings=ent_size, embedding_dim=embed_size)
        self.embeddings_paths = nn.Embedding(num_embeddings=path_size, embedding_dim=embed_size)

        # initialize the word vectors so that the components are normally distributed
        # with mean 0 and variance 0.1
        self.embeddings_entities.weight.data.normal_(mean=0, std=0.1)
        self.embeddings_paths.weight.data.normal_(mean=0, std=0.1)
        self.relu = nn.ReLU()
        # self.rnn = nn.LSTM(input_size=embed_size, hidden_size=100, num_layers=1, batch_first=True)
        # self.rnn2 = nn.LSTM(input_size=100, hidden_size=50, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(300,100)
        self.linear2 = nn.Linear(100,1)
        # self.conv1d = nn.Conv1d(in_channels=2, out_channels=100, kernel_size=1)



    def forward(self, ents_path_idxs):
        pos_idxs= ents_path_idxs[:,1]  # (batch_size,1)
        neg_idxs = ents_path_idxs[:,2]  # (batch_size, 1)
        path_idxs = ents_path_idxs[:, 0]
        pos_vecs = self.embeddings_entities(pos_idxs)
        neg_vecs = self.embeddings_entities(neg_idxs)
        path_vecs = self.embeddings_paths(path_idxs)

        pos_vecs = pos_vecs.unsqueeze_(1)
        neg_vecs = neg_vecs.unsqueeze_(1)
        path_vecs = path_vecs.unsqueeze_(1)

        pos_embeddings = torch.cat((pos_vecs, path_vecs),1)
        neg_embeedings = torch.cat((neg_vecs, path_vecs),1)

        # pos_output = self.conv1d(pos_embeddings) # batch_size, out_channels, embed_size
        # neg_output = self.conv1d(neg_embeedings)


        # pos_output, hn = self.rnn2(pos_output)
        # neg_output, hn = self.rnn2(neg_output)

        pos_output = self.linear2(self.linear1(pos_embeddings))
        neg_output = self.linear2(self.linear1(neg_embeedings))

        self.relu(pos_output)
        self.relu(neg_output)
        pos_output = self.linear2(self.linear1(pos_embeddings))
        neg_output = self.linear2(self.linear1(neg_embeedings))

        self.relu(pos_output)
        self.relu(neg_output)

        diff = pos_output-neg_output

        return diff



