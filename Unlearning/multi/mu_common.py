"""Common code: 4 datasets (MNIST, SVHN, CIFAR-10, CIFAR-100) x 4 models (CNN, MLP, ResNet-18, tiny ViT) on MPS.
All images become uint8 [N,3,32,32] kept on the device; normalisation (x/255-0.5)/0.5 per batch."""
import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
import torchvision

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SPEC = {  # hf id, config, image col, label col, #classes
    "mnist": ("ylecun/mnist", None, "image", "label", 10),
    "svhn": ("ufldl-stanford/svhn", "cropped_digits", "image", "label", 10),
    "cifar10": ("uoft-cs/cifar10", None, "img", "label", 10),
    "cifar100": ("uoft-cs/cifar100", None, "img", "fine_label", 100),
}


def _to_u8(ds, ic, lc):
    xs = []
    for im in ds[ic]:
        a = np.asarray(im.convert("RGB"))
        if a.shape[:2] == (28, 28): a = np.pad(a, ((2, 2), (2, 2), (0, 0)))
        xs.append(a)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(ds[lc])


def load(name):
    hf, cfg, ic, lc, nc = SPEC[name]
    d = load_dataset(hf, cfg) if cfg else load_dataset(hf)
    xtr, ytr = _to_u8(d["train"], ic, lc); xte, yte = _to_u8(d["test"], ic, lc)
    return xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV), nc


def prep(xb): return (xb.float() / 255.0 - 0.5) / 0.5


class CNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.c1 = nn.Conv2d(3, 6, 5); self.c2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(400, 120); self.f2 = nn.Linear(120, 84); self.f3 = nn.Linear(84, nc)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f3(F.relu(self.f2(F.relu(self.f1(x.flatten(1))))))


class MLP(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nc))
    def forward(self, x): return self.net(x)


def resnet18(nc):
    m = torchvision.models.resnet18(num_classes=nc)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()  # CIFAR-style stem
    return m


class ViT(nn.Module):
    """tiny ViT: patch 4 -> 64 tokens, dim 128, depth 4, heads 4."""
    def __init__(self, nc, dim=128, depth=4, heads=4):
        super().__init__()
        self.pe = nn.Conv2d(3, dim, 4, 4); self.cls = nn.Parameter(torch.zeros(1, 1, dim)); self.pos = nn.Parameter(torch.randn(1, 65, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, 0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False); self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, nc)
    def forward(self, x):
        t = self.pe(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t))[:, 0])


MODELS = {"cnn": CNN, "mlp": MLP, "resnet18": resnet18, "vit": ViT}
OPT = {  # same recipe for every dataset
    "cnn": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "mlp": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "resnet18": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=5e-4),
    "vit": lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.05),
}
BS = 128


def make(model, nc, seed):
    torch.manual_seed(seed); return MODELS[model](nc).to(DEV)


def train_idx(net, model, x, y, idx, epochs, seed):
    """epochs over x[idx]; returns sample passes. idx: LongTensor on DEV."""
    opt = OPT[model](net.parameters()); net.train(); n = len(idx)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = idx[torch.randperm(n, generator=g).to(DEV)]
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            if len(b) < 2: continue
            opt.zero_grad(); F.cross_entropy(net(prep(x[b])), y[b]).backward(); opt.step()
    return n * epochs


@torch.no_grad()
def logits_of(net, x, bs=1024):
    net.eval(); return torch.cat([net(prep(x[i:i + bs])) for i in range(0, len(x), bs)])
