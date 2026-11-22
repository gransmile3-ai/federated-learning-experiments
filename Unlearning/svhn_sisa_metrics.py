"""SISA on SVHN measured with the SAME forget sets and metrics as svhn_ga.py (rand 1000 / class 3), S in (5,10), R=5, E=3.
usage: python svhn_sisa_metrics.py <out.json> [seeds=0,1,2]"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from svhn_sisa import Net, tens, shard_train, R
from svhn_ga import metrics, NF, CLS

SS = (5, 10)


class Ens(nn.Module):
    """mean-softmax ensemble; returns log(mean prob) so softmax(forward) == mean prob."""
    def __init__(self, ws):
        super().__init__(); self.ms = nn.ModuleList()
        for w in ws:
            m = Net(); m.load_state_dict(w); self.ms.append(m)
    def forward(self, x):
        return torch.log(torch.stack([F.softmax(m(x), 1) for m in self.ms]).mean(0).clamp_min(1e-12))


def main():
    out = sys.argv[1]; seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
    d = load_dataset("ufldl-stanford/svhn", "cropped_digits")
    x, y = tens(d["train"]); xt, yt = tens(d["test"]); N = len(y)
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["seed"] for r in res}
    for seed in seeds:
        if seed in done: continue
        torch.manual_seed(seed); w_init = {k: v.clone() for k, v in Net().state_dict().items()}
        rng = np.random.RandomState(seed + 13); rec = {"seed": seed, "S": {}}
        # identical forget sets / eval sets as svhn_ga.py
        scen = {}
        for kind in ("rand", "cls"):
            fidx = sorted(int(i) for i in rng.choice(N, NF, replace=False)) if kind == "rand" else [i for i in range(N) if int(y[i]) == CLS]
            fs = set(fidx); ridx = [i for i in range(N) if i not in fs]
            rr = np.random.RandomState(seed + 5); reval = rr.choice(ridx, 5000, replace=False).tolist()
            D = {"kind": kind, "forget": (x[fidx], y[fidx]), "retain_eval": (x[reval], y[reval]), "test": (xt, yt),
                 "test_cls": (xt[yt == CLS], yt[yt == CLS]), "test_other": (xt[yt != CLS], yt[yt != CLS])}
            if kind == "rand":
                hi = rr.choice(len(yt), NF, replace=False); D["heldout"] = (xt[hi], yt[hi])
            scen[kind] = (fidx, D)
        for S in SS:
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7919)).tolist()
            shards = [perm[s::S] for s in range(S)]; where = {}
            for s, sh in enumerate(shards):
                for p, i in enumerate(sh): where[i] = (s, p)
            ck, full_sp, full_s = [], 0, 0.0
            for s in range(S):
                o, sp, secs = shard_train(x, y, shards[s], 1, w_init, seed * 100 + s, f"S{S}"); ck.append(o); full_sp += sp; full_s += secs
            sr = {"full_sample_passes": full_sp}
            for kind, (fidx, D) in scen.items():
                fs = set(fidx); aff = {}
                for i in fidx:
                    s, p = where[i]; sl = p * R // len(shards[s]) + 1; aff[s] = min(aff.get(s, R + 1), sl)
                ws_full = [ck[s][R] for s in range(S)]
                sr[f"{kind}_before"] = metrics(Ens(ws_full), D, "sisa full ensemble", full_sp, full_s)
                ws, sp_tot, secs_tot = [], 0, 0.0
                for s in range(S):
                    if s not in aff: ws.append(ck[s][R]); continue
                    new_order = [i for i in shards[s] if i not in fs]; first = aff[s]
                    o, sp, secs = shard_train(x, y, new_order, first, w_init if first == 1 else ck[s][first - 1], seed * 100 + s + 50, f"S{S}d")
                    ws.append(o[R]); sp_tot += sp; secs_tot += secs
                m = metrics(Ens(ws), D, f"sisa S={S} after unlearn", sp_tot, secs_tot); m["shards_hit"] = len(aff); sr[f"{kind}_after"] = m
                print(f"[seed {seed} S={S} {kind}] shards hit {len(aff)}/{S} cost {sp_tot / 1e6:.2f}M | test {m['test_acc']:.3f} forget {m['forget_acc']:.3f} fcls_test {m['test_forgetclass_acc']:.3f}", flush=True)
            rec["S"][S] = sr
        res.append(rec); json.dump(res, open(out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
