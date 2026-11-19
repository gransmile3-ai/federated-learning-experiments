"""FL client unlearning: forget the desktop client (partition 0) from a FedAvg model.

Same data/model/hparams as ~/FL/NonIID (CIFAR-10, Dirichlet alpha=0.3, 2 clients,
desktop=partition 0, Mac=partition 1). The FedAvg loop is plain PyTorch so that every
round's client update can be stored (needed by FedEraser) and so unlearning runs are cheap.

Methods compared against the gold standard "retrain from scratch without the desktop":
  original   FedAvg with both clients
  retrain    FedAvg from the same init with only the Mac client
  subtract   global - sum of the desktop's stored (weighted) updates
  ga         gradient ascent on the desktop's data (done by the desktop client), no repair
  ga_repair  GA + a few FedAvg rounds on the remaining client
  federaser  FedEraser (Liu et al. 2021): calibrated replay of the Mac client's stored updates
"""
import json, sys, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from fedcompare.task import Net

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
ROUNDS, EPOCHS, LR, BS = 20, 3, 0.01, 32
DEV = "cpu"
torch.set_num_threads(8)
FORGET, KEEP = 0, 1  # desktop, Mac


def to_tensors(ds):
    x = np.stack([np.asarray(i, dtype=np.float32) for i in ds["img"]]) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds["label"])


def load_all():
    part = DirichletPartitioner(num_partitions=2, partition_by="label", alpha=0.3,
                                min_partition_size=1000, self_balancing=False, seed=42)
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": part})
    data = {}
    for pid in (0, 1):
        s = fds.load_partition(pid).train_test_split(test_size=0.2, seed=42)
        data[pid] = {"train": to_tensors(s["train"]), "test": to_tensors(s["test"])}
    data["global_test"] = to_tensors(load_dataset("uoft-cs/cifar10", split="test"))
    return data


def loader(xy, shuffle=False, bs=BS, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*xy), batch_size=bs, shuffle=shuffle, generator=g)


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def setflat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def layer_slices(m):
    s, i = [], 0
    for p in m.parameters():
        s.append((i, i + p.numel())); i += p.numel()
    return s


def local_train(model, xy, epochs, lr=LR, seed=0):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dl = loader(xy, True, seed=seed)
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()


def fedavg(init, clients, data, rounds, tag, record=False):
    """clients: list of partition ids. Returns final flat weights, history, wall time."""
    net = Net(); setflat(net, init.clone())
    sizes = {c: len(data[c]["train"][1]) for c in clients}
    tot = sum(sizes.values())
    hist, t0 = [], time.time()
    for r in range(rounds):
        g = flat(net); ups = {}
        for c in clients:
            m = Net(); setflat(m, g.clone())
            local_train(m, data[c]["train"], EPOCHS, seed=1000 * r + c)
            ups[c] = flat(m) - g
        agg = sum(ups[c] * sizes[c] / tot for c in clients)
        setflat(net, g + agg)
        acc = evaluate(net, data["global_test"])["acc"]
        print(f"[{tag}] round {r + 1}/{rounds} global test acc {acc:.4f}", flush=True)
        hist.append({"round": r + 1, "global": g.clone() if record else None,
                     "updates": ups if record else None, "acc": acc})
    return flat(net), hist, time.time() - t0


@torch.no_grad()
def evaluate(net, xy, bs=512):
    net.eval(); ls, preds = [], []
    for x, y in loader(xy, False, bs):
        o = net(x); ls.append(F.cross_entropy(o, y, reduction="none")); preds.append(o.argmax(1))
    loss = torch.cat(ls); pred = torch.cat(preds)
    return {"acc": (pred == xy[1]).float().mean().item(), "loss": loss.mean().item(),
            "losses": loss.numpy(), "pred": pred.numpy()}


def mia_auc(net, data):
    """Loss-threshold MIA: forget-client train (members) vs the same client's held-out split."""
    m = evaluate(net, data[FORGET]["train"])["losses"]
    n = evaluate(net, data[FORGET]["test"])["losses"]
    k = min(len(m), len(n)); rng = np.random.RandomState(0)
    m = rng.choice(m, k, replace=False); n = rng.choice(n, k, replace=False)
    # AUC of score=-loss for members vs non-members (Mann-Whitney U); low loss => member
    sm, sn = -m, -n
    allv = np.r_[sm, sn]; ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:k].sum() - k * (k + 1) / 2) / (k * k))


def metrics(w, data, name, cost_epochs, secs):
    net = Net(); setflat(net, w)
    g = evaluate(net, data["global_test"])
    y = data["global_test"][1].numpy()
    per_cls = [float((g["pred"][y == c] == c).mean()) for c in range(10)]
    r = {"name": name, "global_test_acc": g["acc"], "per_class_acc": per_cls,
         "forget_train_acc": evaluate(net, data[FORGET]["train"])["acc"],
         "forget_test_acc": evaluate(net, data[FORGET]["test"])["acc"],
         "retain_train_acc": evaluate(net, data[KEEP]["train"])["acc"],
         "retain_test_acc": evaluate(net, data[KEEP]["test"])["acc"],
         "mia_auc": mia_auc(net, data), "local_epoch_cost": cost_epochs, "seconds": secs}
    print(f"== {name}: test {r['global_test_acc']:.4f} forget_train {r['forget_train_acc']:.4f} "
          f"retain_test {r['retain_test_acc']:.4f} MIA {r['mia_auc']:.3f} cost {cost_epochs} ep {secs:.0f}s", flush=True)
    return r


def gradient_ascent(w, data, steps, lr, clip=None):
    net = Net(); setflat(net, w.clone()); net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr)  # no momentum: keeps ascent stable
    dl = loader(data[FORGET]["train"], True, seed=7); it = iter(dl)
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(dl); x, y = next(it)
        opt.zero_grad(); (-F.cross_entropy(net(x), y)).backward()
        if clip: torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
    return flat(net)


def federaser(hist, init, data, dt=2, e_cal=2):
    """Calibrated replay with the remaining (Mac) client only. Per-layer: keep the direction of
    the stored old update (summed over the dt-round interval), take the norm from a new short
    local training run starting at the calibrated model."""
    net = Net(); sl = layer_slices(net); w = init.clone(); epochs_used = 0
    for s in range(0, len(hist), dt):
        old = sum(hist[r]["updates"][KEEP] for r in range(s, min(s + dt, len(hist))))
        m = Net(); setflat(m, w.clone())
        local_train(m, data[KEEP]["train"], e_cal, seed=5000 + s); epochs_used += e_cal
        new = flat(m) - w
        cal = torch.zeros_like(w)
        for a, b in sl:
            cal[a:b] = new[a:b].norm() * old[a:b] / (old[a:b].norm() + 1e-12)
        w = w + cal
    return w, epochs_used


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    data = load_all()
    for pid in (0, 1):
        print("partition", pid, "train", len(data[pid]["train"][1]),
              "label hist", torch.bincount(data[pid]["train"][1], minlength=10).tolist(), flush=True)
    torch.manual_seed(42); init = flat(Net())
    res = {}
    # 1) original FL (both clients), keep every round's updates
    w_orig, hist, t = fedavg(init, [FORGET, KEEP], data, ROUNDS, "original", record=True)
    res["original"] = metrics(w_orig, data, "original", ROUNDS * EPOCHS * 2, t)
    res["original"]["curve"] = [h["acc"] for h in hist]
    torch.save({"w": w_orig}, f"{OUT}/original.pt")
    # 2) retrain without the desktop (gold standard)
    w_re, hist_re, t = fedavg(init, [KEEP], data, ROUNDS, "retrain")
    res["retrain"] = metrics(w_re, data, "retrain", ROUNDS * EPOCHS, t)
    res["retrain"]["curve"] = [h["acc"] for h in hist_re]
    torch.save({"w": w_re}, f"{OUT}/retrain.pt")
    # 3) subtract the desktop's weighted updates
    n0, n1 = (len(data[c]["train"][1]) for c in (FORGET, KEEP))
    sub = w_orig - sum(h["updates"][FORGET] * n0 / (n0 + n1) for h in hist)
    res["subtract"] = metrics(sub, data, "subtract", 0, 0.0)
    # 4) gradient ascent sweep (cost = steps / batches-per-epoch of the desktop client)
    bpe = len(data[FORGET]["train"][1]) / BS
    res["ga_sweep"] = []
    for lr in (1e-3, 5e-3):
        for steps in (25, 50, 100, 200, 400):
            t0 = time.time(); w = gradient_ascent(w_orig, data, steps, lr)
            r = metrics(w, data, f"ga lr={lr} steps={steps}", steps / bpe, time.time() - t0)
            r.update(lr=lr, steps=steps); res["ga_sweep"].append(r)
    # 5) FedEraser
    t0 = time.time(); w_fe, ep = federaser(hist, init, data)
    res["federaser"] = metrics(w_fe, data, "federaser", ep, time.time() - t0)
    torch.save({"w": w_fe}, f"{OUT}/federaser.pt")
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    torch.save({"hist": [{"round": h["round"], "global": h["global"], "updates": h["updates"]} for h in hist]},
               f"{OUT}/orig_history.pt")
    print("saved stage 1", flush=True)


if __name__ == "__main__":
    main()
