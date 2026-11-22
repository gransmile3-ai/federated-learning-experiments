import time, sys, torch
import mu_common as M
x, y, xt, yt, nc = M.load("cifar10"); print("loaded", x.shape, M.DEV, flush=True)
idx = torch.arange(len(y), device=M.DEV)[:10000]
for m in ("cnn", "mlp", "resnet18", "vit"):
    net = M.make(m, nc, 0); M.train_idx(net, m, x, y, idx[:1000], 1, 0); torch.mps.synchronize()
    t = time.time(); M.train_idx(net, m, x, y, idx, 1, 0); torch.mps.synchronize()
    dt = time.time() - t
    lg = M.logits_of(net, xt); acc = (lg.argmax(1) == yt).float().mean().item()
    print(f"{m}: {dt:.1f}s per 10k samples -> {dt*5:.0f}s per 50k epoch (test acc after 1 ep on 10k: {acc:.3f})", flush=True)
