"""SISA on SVHN (dataset used in Bourtoule et al. 2021), run on one machine.
Compare S in (5,10,20) shards, R=5 slices, 3 seeds, vs naive retraining with an equal budget.
Budget: SISA full training = sum_r (n_shard*r/R)*E sample passes per shard = 9N for E=3, R=5;
naive single model = 9 epochs on all N -> equal sample-pass budget.
Deletion scenarios: 1 random sample, 10 random, 100 random (uniform requests = SISA's worst case).
Cost = sample passes actually retrained (SISA: only affected shards, from the checkpoint before the first affected slice).
usage: python svhn_sisa.py <out.json> [seeds=0,1,2]
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset

R, LR, BS = 5, 0.01, 64
E = int(os.environ.get("SISA_E", 3))
SHARDS = tuple(int(s) for s in os.environ.get("SISA_S", "5,10,20").split(","))
KS = (1, 10, 100)
torch.set_num_threads(os.cpu_count() or 4)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5); self.pool = nn.MaxPool2d(2, 2); self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120); self.fc2 = nn.Linear(120, 84); self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))); x = self.pool(F.relu(self.conv2(x))); x = x.reshape(-1, 400)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def tens(ds):
    x = np.stack([np.asarray(i.convert("RGB"), dtype=np.float32) for i in ds["image"]]) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds["label"])


def train(net, x, y, idx, epochs, seed):
    dl = DataLoader(TensorDataset(x[idx], y[idx]), batch_size=BS, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9); net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    return len(idx) * epochs


@torch.no_grad()
def probs(net, x):
    net.eval(); return torch.cat([F.softmax(net(x[i:i + 4096]), 1) for i in range(0, len(x), 4096)])


def acc_of(p, y): return (p.argmax(1) == y).float().mean().item()


def shard_train(x, y, order, start, w0, seed, tag):
    """slices start..R on cumulative prefixes of `order`; returns ({r: state}, sample_passes, seconds)."""
    n = len(order); net = Net(); net.load_state_dict(w0); outs = {}; sp = 0; t0 = time.time()
    for r in range(start, R + 1):
        sp += train(net, x, y, order[: n * r // R], E, seed * 1000 + r)
        outs[r] = {k: v.clone() for k, v in net.state_dict().items()}
    return outs, sp, time.time() - t0


def main():
    out = sys.argv[1]; seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
    d = load_dataset("ufldl-stanford/svhn", "cropped_digits")
    x, y = tens(d["train"]); xt, yt = tens(d["test"]); N = len(y)
    print("train", N, "test", len(yt), flush=True)
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["seed"] for r in res}
    for seed in seeds:
        if seed in done: continue
        rec = {"seed": seed, "N": N, "R": R, "E": E}
        # naive baseline with equal budget (9 epochs on everything) and its retrain after deleting K points
        torch.manual_seed(seed); w_init = {k: v.clone() for k, v in Net().state_dict().items()}
        net = Net(); net.load_state_dict(w_init); t0 = time.time(); sp = train(net, x, y, list(range(N)), 9, seed)
        rec["naive"] = {"test_acc": acc_of(probs(net, xt), yt), "sample_passes": sp, "seconds": time.time() - t0}
        print(f"[seed {seed}] naive test {rec['naive']['test_acc']:.4f} {rec['naive']['seconds']:.0f}s", flush=True)
        rng = np.random.RandomState(seed + 13)
        dels = {k: sorted(int(i) for i in rng.choice(N, k, replace=False)) for k in KS}
        rec["naive_del"] = {}
        for k in KS:
            keep = [i for i in range(N) if i not in set(dels[k])]
            net = Net(); net.load_state_dict(w_init); t0 = time.time(); sp = train(net, x, y, keep, 9, seed)
            rec["naive_del"][k] = {"test_acc": acc_of(probs(net, xt), yt), "sample_passes": sp, "seconds": time.time() - t0}
        rec["sisa"] = {}
        for S in SHARDS:
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7919)).tolist()  # own seed
            shards = [perm[s::S] for s in range(S)]             # shard s holds perm[s::S]; order inside = slice order
            where = {}                                          # sample -> (shard, position)
            for s, sh in enumerate(shards):
                for p, i in enumerate(sh): where[i] = (s, p)
            ck, fin, full_sp, full_s = [], [], 0, 0.0
            for s in range(S):
                o, sp, secs = shard_train(x, y, shards[s], 1, w_init, seed * 100 + s, f"S{S}")
                ck.append(o); full_sp += sp; full_s += secs
            P = sum(probs(_load(ck[s][R]), xt) for s in range(S)) / S
            sr = {"full": {"test_acc": acc_of(P, yt), "sample_passes": full_sp, "seconds": full_s}, "del": {}}
            print(f"[seed {seed}] S={S} ensemble test {sr['full']['test_acc']:.4f} full {full_s:.0f}s", flush=True)
            for k in KS:
                dset = set(dels[k]); sp_tot, secs_tot = 0, 0.0; ens = []
                aff = {}
                for i in dels[k]:
                    s, p = where[i]; sl = p * R // len(shards[s]) + 1
                    aff[s] = min(aff.get(s, R + 1), sl)
                for s in range(S):
                    if s not in aff: ens.append(ck[s][R]); continue
                    new_order = [i for i in shards[s] if i not in dset]; first = aff[s]
                    w0 = w_init if first == 1 else ck[s][first - 1]
                    o, sp, secs = shard_train(x, y, new_order, first, w0, seed * 100 + s + 50, f"S{S}d")
                    ens.append(o[R]); sp_tot += sp; secs_tot += secs
                Pd = sum(probs(_load(w), xt) for w in ens) / S
                sr["del"][k] = {"test_acc": acc_of(Pd, yt), "sample_passes": sp_tot, "seconds": secs_tot,
                                "shards_hit": len(aff), "cost_vs_full": sp_tot / full_sp,
                                "speedup_vs_naive": rec["naive_del"][k]["sample_passes"] / max(sp_tot, 1)}
                print(f"  K={k}: shards hit {len(aff)}/{S} cost {sp_tot / full_sp:.2f} of SISA-full, test {sr['del'][k]['test_acc']:.4f}", flush=True)
            rec["sisa"][S] = sr
        res.append(rec); json.dump(res, open(out, "w"), indent=1)
    print("done", flush=True)


def _load(w):
    m = Net(); m.load_state_dict(w); return m


if __name__ == "__main__":
    main()
