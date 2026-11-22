"""Gradient-ascent unlearning on SVHN (same data/CNN as svhn_sisa.py). Single central model, Mac.
Scenarios: (rand) forget 1000 random train samples; (cls) forget every training sample of digit 3.
Gold standard: retrain from the same init on the remaining data (9 epochs).
GA: plain SGD (no momentum) ascent on forget batches. Reports (a) fixed-step trajectory, (b) stop at forget loss >= ln10,
(c) stop + short repair: 1 epoch of fine-tuning on 5000 random retain samples.
usage: python svhn_ga.py <out.json> [seeds=0,1,2]
"""
import json, math, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from svhn_sisa import Net, tens, train, probs, BS

LN10 = math.log(10); CLS = 3; NF = 1000; REPAIR_N = 5000
LRS = (1e-3, 3e-3, 1e-2); MAXSTEP = 300; CHECK = 5
torch.set_num_threads(os.cpu_count() or 4)


@torch.no_grad()
def ev(net, x, y):
    p = probs(net, x); loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), y, reduction="none")
    return (p.argmax(1) == y).float().mean().item(), loss.mean().item(), loss.numpy()


def auc(mem, non):
    k = min(len(mem), len(non)); rng = np.random.RandomState(0)
    m = rng.choice(mem, k, replace=False); n = rng.choice(non, k, replace=False)
    ranks = np.r_[-m, -n].argsort().argsort() + 1.0
    return float((ranks[:k].sum() - k * (k + 1) / 2) / (k * k))


def metrics(net, D, name, cost, secs):
    fa, fl, flv = ev(net, *D["forget"]); ra, _, _ = ev(net, *D["retain_eval"]); ta, _, tlv = ev(net, *D["test"])
    ca, _, clv = ev(net, *D["test_cls"]); oa, _, olv = ev(net, *D["test_other"])
    r = {"name": name, "test_acc": ta, "forget_acc": fa, "forget_loss": fl, "retain_acc": ra,
         "test_forgetclass_acc": ca, "test_otherclass_acc": oa, "cost_sample_passes": cost, "seconds": secs}
    if D["kind"] == "rand": r["mia_auc"] = auc(flv, ev(net, *D["heldout"])[2])
    return r


def ga(w0, D, lr, seed, rule):
    """ascent until rule(net) true or MAXSTEP; returns net, steps, trajectory"""
    net = Net(); net.load_state_dict(w0); net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    fx, fy = D["forget"]; dl = DataLoader(TensorDataset(fx, fy), batch_size=BS, shuffle=True, generator=torch.Generator().manual_seed(seed))
    it = iter(dl); traj = []; steps = 0
    sub = (fx[:512], fy[:512])
    while steps < MAXSTEP:
        try: xb, yb = next(it)
        except StopIteration: it = iter(dl); xb, yb = next(it)
        opt.zero_grad(); (-F.cross_entropy(net(xb), yb)).backward(); opt.step(); steps += 1
        if steps % CHECK == 0:
            fa, fl, _ = ev(net, *sub); net.train()
            ta, _, _ = ev(net, *D["test"]); net.train()
            traj.append({"step": steps, "forget_loss": fl, "forget_acc": fa, "test_acc": ta})
            if fl != fl or fl > 1e4: break
            if rule is not None and rule(fl): break
    return net, steps, traj


def main():
    out = sys.argv[1]; seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
    d = load_dataset("ufldl-stanford/svhn", "cropped_digits")
    x, y = tens(d["train"]); xt, yt = tens(d["test"]); N = len(y)
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["seed"] for r in res}
    for seed in seeds:
        if seed in done: continue
        torch.manual_seed(seed); w_init = {k: v.clone() for k, v in Net().state_dict().items()}
        net = Net(); net.load_state_dict(w_init); t0 = time.time(); train(net, x, y, list(range(N)), 9, seed)
        w_orig = {k: v.clone() for k, v in net.state_dict().items()}; t_orig = time.time() - t0
        rng = np.random.RandomState(seed + 13)
        rec = {"seed": seed, "orig_seconds": t_orig, "scen": {}}
        for kind in ("rand", "cls"):
            if kind == "rand":
                fidx = sorted(int(i) for i in rng.choice(N, NF, replace=False))
                held = None
            else:
                fidx = [i for i in range(N) if int(y[i]) == CLS]
            fs = set(fidx); ridx = [i for i in range(N) if i not in fs]
            rr = np.random.RandomState(seed + 5); reval = rr.choice(ridx, 5000, replace=False).tolist()
            D = {"kind": kind, "forget": (x[fidx], y[fidx]), "retain_eval": (x[reval], y[reval]), "test": (xt, yt),
                 "test_cls": (xt[yt == CLS], yt[yt == CLS]), "test_other": (xt[yt != CLS], yt[yt != CLS])}
            if kind == "rand":
                # held-out non-members: a test-set split of the same size as the forget set (MIA reference)
                hi = rr.choice(len(yt), NF, replace=False); D["heldout"] = (xt[hi], yt[hi])
            S = {}
            n0 = Net(); n0.load_state_dict(w_orig)
            S["original"] = metrics(n0, D, "original", 9 * N, t_orig)
            nr = Net(); nr.load_state_dict(w_init); t0 = time.time(); sp = train(nr, x, y, ridx, 9, seed)
            S["retrain"] = metrics(nr, D, "retrain", sp, time.time() - t0)
            S["sweep"] = []; S["stop"] = []; S["stop_repair"] = []
            for lr in LRS:
                # (a) trajectory without stopping (logs collapse)
                _, st, traj = ga(w_orig, D, lr, seed, None)
                S["sweep"].append({"lr": lr, "traj": traj})
                # (b) stop rule: forget loss >= ln10
                t0 = time.time(); g, st, _ = ga(w_orig, D, lr, seed, lambda fl: fl >= LN10)
                secs = time.time() - t0
                m = metrics(g, D, f"ga lr={lr} stop", st * BS, secs); m.update(lr=lr, steps=st); S["stop"].append(m)
                # (c) + repair on a small retain subset
                rg = np.random.RandomState(seed + 99).choice(ridx, REPAIR_N, replace=False).tolist()
                t1 = time.time(); sp = train(g, x, y, rg, 1, seed + 77)
                m = metrics(g, D, f"ga lr={lr} stop+repair", st * BS + sp, secs + time.time() - t1); m.update(lr=lr, steps=st); S["stop_repair"].append(m)
                print(f"[seed {seed} {kind}] lr {lr}: steps {st} stop test {S['stop'][-1]['test_acc']:.3f} forget {S['stop'][-1]['forget_acc']:.3f}"
                      f" | repair test {m['test_acc']:.3f} forget {m['forget_acc']:.3f} forgetclass-test {m['test_forgetclass_acc']:.3f}", flush=True)
            print(f"[seed {seed} {kind}] orig test {S['original']['test_acc']:.3f} forget {S['original']['forget_acc']:.3f} | retrain test {S['retrain']['test_acc']:.3f} forget {S['retrain']['forget_acc']:.3f}", flush=True)
            rec["scen"][kind] = S
        res.append(rec); json.dump(res, open(out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
