import torch.nn as nn
import torch

import config


class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dims=None, padding_idx=0):
        super(NBoW, self).__init__()
        if hidden_dims is None:
            hidden_dims = [output_dim]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        layers = [nn.Linear(embedding_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, sentence_length)
        x = self.embedding(x)
        # x: (batch_size, sentence_length, embedding_size)
        x = x.mean(dim=1)
        # x: (batch_size, embedding_size)
        return self.net(x)  # (batch_size, output_size)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim,
                 lstm_hidden_dims=300,
                 lstm_layers=2,
                 hidden_dims=None,
                 padding_idx=0,
                 bidirectional=False):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional

        if hidden_dims is None:
            hidden_dims = [output_dim]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dims, lstm_layers, bidirectional=bidirectional, batch_first=True, dropout=config.DROPOUT)
        layers = [nn.Linear(lstm_hidden_dims * (2 if bidirectional else 1), hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = (batch_size, sentence_length)
        x = self.embedding(x)
        # x = (batch_size, sentence_length, embedding_size)
        _, (x, _) = self.lstm(x)
        if self.bidirectional:
            x = torch.cat([x[-1], x[-2]], dim=-1)
        else:
            x = x[-1]
        return self.net(x)
