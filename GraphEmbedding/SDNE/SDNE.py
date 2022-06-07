from torch import nn


class SDNE_model(nn.Module):
    def __init__(self, node_size, hidden_size=[256, 128], **kwargs):
        super(SDNE_model, self).__init__(**kwargs)
        self.encoders = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                self.encoders.add_module(f"encoder{i}", nn.Linear(node_size, hidden_size[i]))
                self.encoders.add_module(f"relu{i}", nn.ReLU())
            else:
                self.encoders.add_module(f"encoder{i}", nn.Linear(hidden_size[i - 1], hidden_size[i]))
                self.encoders.add_module(f"relu{i}", nn.ReLU())
        self.decoders = nn.Sequential()
        for i in reversed(range(len(hidden_size))):
            if i == 0:
                self.decoders.add_module(f"decoder{i}", nn.Linear(hidden_size[i], node_size))
                self.decoders.add_module(f"relu{i}", nn.ReLU())
            else:
                self.decoders.add_module(f"decoder{i}", nn.Linear(hidden_size[i], hidden_size[i - 1]))
                self.decoders.add_module(f"relu{i}", nn.ReLU())

    def forward(self, X_input):
        for encoder in self.encoders:
            X_input = encoder(X_input)
        emb = X_input
        for decoder in self.decoders:
            X_input = decoder(X_input)
        return X_input, emb
