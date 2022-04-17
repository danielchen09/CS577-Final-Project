import torch.nn as nn


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
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.net(x)