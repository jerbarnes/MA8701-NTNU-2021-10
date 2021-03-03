import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence, pack_sequence

class LSTM_Model(nn.Module):

    def __init__(self,
                 word2idx,
                 embedding_dim=50,
                 hidden_dim=50,
                 num_labels=2):
        super(LSTM_Model, self).__init__()
        self.word2idx = word2idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # create a randomly initialized embedding matrix, where each entry
        # in the vocabulary is an n-dimensional entry which will be updated
        # when training
        self.embedding = nn.Embedding(len(word2idx), self.embedding_dim)

        # a single layered forward LSTM which will return a hidden state for a sentence
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=False)

        # include a dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

        # this final linear classifier maps the final state of the LSTM to the label space
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

    def max_pool(self, x):
        # embed the word indices and create a packedsequence to pass to LSTM
        emb = self.embedding(x.data)
        packed_emb = PackedSequence(emb, x.batch_sizes)
        # the output is a packed version of all of the outputs of the LSTM at each timestep
        output, _ = self.lstm(packed_emb)
        # unpack this representation
        o, _ = pad_packed_sequence(output, batch_first=True)
        # we take the max of these output representations over all the timesteps
        o = self.dropout(o)
        o, _ = o.max(dim=1)
        # project this to the label space
        o = self.classifier(o)
        return o

    def predict_text(self, text):
        label_map = {1: "Positive", 0: "Negative"}
        idxs = torch.LongTensor(self.word2idx.ws2ids(text.split()))
        x = pack_sequence(idxs.unsqueeze(0))
        logits = self.max_pool(x)
        _, preds = logits.max(1)
        return label_map[int(preds[0])]

