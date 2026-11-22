"""One (dataset, model, seed) combo: retrain / GA / GA+repair / SISA(S=5) on two forget targets.
usage: python multi_run.py <dataset> <model> <seed> <out.json>
Recipe (identical for every combo): 6 epochs total. SISA: S=5 shards, R=5 slices, E=2 epochs per cumulative slice (=6 epoch-equivalents).
Forget targets: 'rand' = 1000 random train samples, 'cls' = every train sample of class 3.
GA: plain SGD ascent (model in eval mode), stops when forget loss >= ln(#classes) (cap 300 steps); lr in {1e-3, 3e-3}.
Repair: 1 epoch on 5000 random retain samples with the model's normal optimizer. Cost = sample passes."""
import json, math, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
import mu_common as M

R, E_SLICE, T_EPOCHS, S = 5, 2, 6, 5
NF, CLS, REPAIR_N, MAXSTEP, GA_LRS = 1000, 3, 5000, 300, (1e-3, 3e-3)
dev = M.DEV


def tidx(l): return torch.as_tensor(l, device=dev, dtype=torch.long)


def probs_ens(model, nc, states, X):
    scratch = M.make(model, nc, 0); acc = 0
    for st in states:
        scratch.load_state_dict(st); acc = acc + F.softmax(M.logits_of(scratch, X), 1)
    return acc / len(states)


def auc(mem, non):
    k = min(len(mem), len(non)); rng = np.random.RandomState(0)
    m = rng.choice(mem, k, replace=False); n = rng.choice(non, k, replace=False)
    ranks = np.r_[-m, -n].argsort().argsort() + 1.0
    return float((ranks[:k].sum() - k * (k + 1) / 2) / (k * k))


class Eval:
    """Probabilities on [forget | retain_eval | test] computed once per model."""
    def __init__(self, x, y, xt, yt, fidx, reval, hidx, kind):
        self.X = torch.cat([x[fidx], x[reval], xt]); self.kind = kind
        self.nf, self.nr = len(fidx), len(reval)
        self.yf, self.yr, self.yt = y[fidx], y[reval], yt; self.hidx = hidx
    def metrics(self, model, nc, states, name, cost, secs):
        P = probs_ens(model, nc, states, self.X)
        pf, pr, pt = P[:self.nf], P[self.nf:self.nf + self.nr], P[self.nf + self.nr:]
        acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
        nll = lambda p, y: (-torch.log(p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12))).cpu().numpy()
        c = self.yt == CLS
        r = {"name": name, "test_acc": acc(pt, self.yt), "forget_acc": acc(pf, self.yf), "retain_acc": acc(pr, self.yr),
             "test_forgetclass_acc": acc(pt[c], self.yt[c]), "test_otherclass_acc": acc(pt[~c], self.yt[~c]),
             "forget_loss": float(nll(pf, self.yf).mean()), "cost_sample_passes": cost, "seconds": secs}
        if self.kind == "rand": r["mia_auc"] = auc(nll(pf, self.yf), nll(pt[self.hidx], self.yt[self.hidx]))
        return r


def sd(net): return {k: v.detach().clone() for k, v in net.state_dict().items()}


def ga(model, nc, w0, x, y, fidx, lr, seed, stop_loss, evalsub):
    net = M.make(model, nc, 0); net.load_state_dict(w0); net.eval()   # eval mode: BN uses running stats
    opt = torch.optim.SGD(net.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    n = len(fidx); order = fidx[torch.randperm(n, generator=g).to(dev)]; pos = 0; steps = 0
    while steps < MAXSTEP:
        if pos + M.BS > n: order = fidx[torch.randperm(n, generator=g).to(dev)]; pos = 0
        b = order[pos:pos + M.BS]; pos += M.BS
        opt.zero_grad(); (-F.cross_entropy(net(M.prep(x[b])), y[b])).backward(); opt.step(); steps += 1
        if steps % 5 == 0:
            with torch.no_grad(): fl = F.cross_entropy(net(M.prep(x[evalsub])), y[evalsub]).item()
            if fl != fl or fl >= stop_loss: break
    return net, steps


def main():
    ds, model, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    t00 = time.time()
    x, y, xt, yt, nc = M.load(ds); N = len(y)
    w_init = sd(M.make(model, nc, seed))
    rec = {"dataset": ds, "model": model, "seed": seed, "N": N, "nc": nc, "scen": {}}
    # original
    net = M.make(model, nc, seed); t0 = time.time(); sp = M.train_idx(net, model, x, y, torch.arange(N, device=dev), T_EPOCHS, seed)
    w_orig = sd(net); rec["orig_seconds"] = time.time() - t0
    # SISA full training (shards share w_init)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7919))
    shards = [perm[s::S].tolist() for s in range(S)]; where = {}
    for s, sh in enumerate(shards):
        for p, i in enumerate(sh): where[i] = (s, p)
    def shard_train(order, start, w0, sseed):
        n = len(order); m = M.make(model, nc, 0); m.load_state_dict(w0); outs = {}; sp = 0
        for r in range(start, R + 1):
            sp += M.train_idx(m, model, x, y, tidx(order[: n * r // R]), E_SLICE, sseed * 1000 + r); outs[r] = sd(m)
        return outs, sp
    ck, full_sp = [], 0; t0 = time.time()
    for s in range(S):
        o, sp = shard_train(shards[s], 1, w_init, seed * 100 + s); ck.append({k: {a: b.cpu() for a, b in v.items()} for k, v in o.items()}); full_sp += sp
    rec["sisa_full_seconds"] = time.time() - t0
    rng = np.random.RandomState(seed + 13)
    for kind in ("rand", "cls"):
        if kind == "rand": fl = sorted(int(i) for i in rng.choice(N, NF, replace=False))
        else: fl = torch.nonzero(y == CLS).squeeze(1).tolist()
        fs = set(fl); ridx = [i for i in range(N) if i not in fs]
        rr = np.random.RandomState(seed + 5); reval = rr.choice(ridx, 5000, replace=False).tolist()
        hidx = rr.choice(len(yt), NF, replace=False) if kind == "rand" else None
        fidx = tidx(fl); ev = Eval(x, y, xt, yt, fidx, tidx(reval), hidx, kind)
        evalsub = fidx[:512]; S_ = {}
        S_["original"] = ev.metrics(model, nc, [w_orig], "original", T_EPOCHS * N, rec["orig_seconds"])
        net = M.make(model, nc, seed); net.load_state_dict(w_init); t0 = time.time()
        sp = M.train_idx(net, model, x, y, tidx(ridx), T_EPOCHS, seed)
        S_["retrain"] = ev.metrics(model, nc, [sd(net)], "retrain", sp, time.time() - t0)
        S_["ga_stop"], S_["ga_repair"] = [], []
        rg = tidx(np.random.RandomState(seed + 99).choice(ridx, REPAIR_N, replace=False).tolist())
        for lr in GA_LRS:
            t0 = time.time(); g, st = ga(model, nc, w_orig, x, y, fidx, lr, seed, math.log(nc), evalsub); secs = time.time() - t0
            m = ev.metrics(model, nc, [sd(g)], f"ga lr={lr}", st * M.BS, secs); m.update(lr=lr, steps=st); S_["ga_stop"].append(m)
            t1 = time.time(); sp = M.train_idx(g, model, x, y, rg, 1, seed + 77)
            m = ev.metrics(model, nc, [sd(g)], f"ga+repair lr={lr}", st * M.BS + sp, secs + time.time() - t1); m.update(lr=lr, steps=st); S_["ga_repair"].append(m)
        # SISA: before / after
        S_["sisa_before"] = ev.metrics(model, nc, [{k: v.to(dev) for k, v in ck[s][R].items()} for s in range(S)], "sisa before", full_sp, rec["sisa_full_seconds"])
        aff = {}
        for i in fl:
            s, p = where[i]; sl = p * R // len(shards[s]) + 1; aff[s] = min(aff.get(s, R + 1), sl)
        ws, sp_tot, t0 = [], 0, time.time()
        for s in range(S):
            if s not in aff: ws.append({k: v.to(dev) for k, v in ck[s][R].items()}); continue
            order = [i for i in shards[s] if i not in fs]; first = aff[s]
            w0 = w_init if first == 1 else {k: v.to(dev) for k, v in ck[s][first - 1].items()}
            o, sp = shard_train(order, first, w0, seed * 100 + s + 50); ws.append(o[R]); sp_tot += sp
        m = ev.metrics(model, nc, ws, "sisa after", sp_tot, time.time() - t0); m["shards_hit"] = len(aff); S_["sisa_after"] = m
        rec["scen"][kind] = S_
        f = lambda k: S_[k]["forget_acc"]; t = lambda k: S_[k]["test_acc"]
        print(f"[{ds}/{model}/s{seed}/{kind}] orig t{t('original'):.3f} f{f('original'):.3f} | retrain t{t('retrain'):.3f} f{f('retrain'):.3f} | "
              f"ga t{S_['ga_stop'][0]['test_acc']:.3f} f{S_['ga_stop'][0]['forget_acc']:.3f} | ga+rep t{S_['ga_repair'][0]['test_acc']:.3f} f{S_['ga_repair'][0]['forget_acc']:.3f} | "
              f"sisa t{t('sisa_after'):.3f} f{f('sisa_after'):.3f}", flush=True)
    rec["total_seconds"] = time.time() - t00
    json.dump(rec, open(out, "w"), indent=1); print("saved", out, f"{rec['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
