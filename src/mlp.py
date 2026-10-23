"""Feed-forward network (MLP) for artwork style classification, built from nn primitives."""
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "gelu": nn.GELU}


class ArtMLP(nn.Module):
    def __init__(self, in_dim=3 * 64 * 64, hidden=(512, 256, 128), n_classes=6,
                 activation="relu", init="he", batchnorm=True, dropout=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        self.linears = []
        for h in hidden:
            lin = nn.Linear(d, h); self.linears.append(lin); layers.append(lin)
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))          # Linear -> BN -> activation
            layers.append(ACTS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        head = nn.Linear(d, n_classes); self.linears.append(head); layers.append(head)
        self.net = nn.Sequential(*layers)               # raw logits -> CrossEntropyLoss
        self._init(init)

    def _init(self, init):
        for lin in self.linears:
            if init == "he":       # Var(W) = 2 / fan_in
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            elif init == "xavier": # Var(W) = 2 / (fan_in + fan_out)
                nn.init.xavier_normal_(lin.weight)
            elif init == "small":
                nn.init.normal_(lin.weight, std=1e-3)
            elif init == "large":  # deliberately bad: signal explosion
                nn.init.normal_(lin.weight, std=1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
