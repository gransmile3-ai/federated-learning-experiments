"""Evaluate SISA (shard=client) models on the Mac. Run with PYTHONPATH=~/FL/NonIID:~/FL/Unlearning."""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import unlearn as U
from sisa_shard import Net as SNet, R

BASE = "/Users/yjj/FL/Unlearning/results_sisa"
data = U.load_all()


class Ens(nn.Module):
    """Mean-softmax ensemble; forward returns log of mean probs so cross_entropy = NLL of the ensemble."""
    def __init__(self, ws):
        super().__init__(); self.ms = nn.ModuleList()
        for w in ws:
            m = SNet(); m.load_state_dict(torch.load(w)); m.eval(); self.ms.append(m)
    def forward(self, x):
        p = torch.stack([F.softmax(m(x), 1) for m in self.ms]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def auc(mem, non):
    k = min(len(mem), len(non)); rng = np.random.RandomState(0)
    m = rng.choice(mem, k, replace=False); n = rng.choice(non, k, replace=False)
    ranks = np.r_[-m, -n].argsort().argsort() + 1.0
    return float((ranks[:k].sum() - k * (k + 1) / 2) / (k * k))


def row(net, dele=None):
    g = U.evaluate(net, data["global_test"]); r = {
        "test_acc": g["acc"], "desktop_train_acc": U.evaluate(net, data[0]["train"])["acc"],
        "mac_test_acc": U.evaluate(net, data[1]["test"])["acc"], "mia_client": U.mia_auc(net, data)}
    if dele is not None:
        xy = (data[0]["train"][0][dele], data[0]["train"][1][dele])
        e = U.evaluate(net, xy); r["del_acc"] = e["acc"]; r["del_loss"] = e["loss"]
        r["mia_del"] = auc(e["losses"], U.evaluate(net, data[0]["test"])["losses"]) if len(dele) > 20 else None
    return r


out = []
for seed in (0, 1, 2):
    d, m = f"{BASE}/desktop/seed{seed}", f"{BASE}/mac/seed{seed}"
    idx = json.load(open(f"{d}/deleted_idx.json")); td = json.load(open(f"{d}/timing.json"))
    tm = json.load(open(f"{m}/timing.json"))
    s = {"seed": seed, "idx_info": {k: idx[k] for k in ("single_slice", "5pct_min_slice")}}
    for nm in ("single", "5pct"):
        s[f"cost_{nm}"] = td[f"del_{nm}_samples"] / td["full_samples"]
    s["desktop_full_s"], s["mac_full_s"] = td["full_s"], tm["full_s"]
    s["full_ens"] = row(Ens([f"{d}/final.pt", f"{m}/final.pt"]), idx["5pct"])
    s["drop_desktop"] = row(Ens([f"{m}/final.pt"]), idx["5pct"])
    s["desktop_only"] = row(Ens([f"{d}/final.pt"]))
    s["ens_del_single"] = row(Ens([f"{d}/del_single.pt", f"{m}/final.pt"]), idx["single"])
    s["ens_del_5pct"] = row(Ens([f"{d}/del_5pct.pt", f"{m}/final.pt"]), idx["5pct"])
    s["full_ens_on_single"] = row(Ens([f"{d}/final.pt", f"{m}/final.pt"]), idx["single"])
    out.append(s)
    print(f"seed {seed}: ens {s['full_ens']['test_acc']:.3f} drop_desktop {s['drop_desktop']['test_acc']:.3f} "
          f"desktop_only {s['desktop_only']['test_acc']:.3f} | 5pct del_acc {s['full_ens']['del_acc']:.3f}->{s['ens_del_5pct']['del_acc']:.3f} "
          f"mia_del {s['full_ens']['mia_del']:.3f}->{s['ens_del_5pct']['mia_del']:.3f} cost1 {s['cost_single']:.2f}", flush=True)
json.dump(out, open(f"{BASE}/sisa_results.json", "w"), indent=1)
