"""EECC table (Mercuri et al. 2022, Sec. 3) from the sweep JSONs.
Efficiency      = time(naive retrain) / time(unlearning)      (also sample-pass ratio)
Effectiveness   = |test acc(unlearned) - test acc(retrain)|     (smaller is better)
Certifiability  = SAPE of acc on D_u: |M_u^U - M_u^*| / (|M_u^U| + |M_u^*|) * 100   (0/0 -> 0)
Consistency     = agreement with retrain predictions on test (higher), KL(retrain||unlearned) (lower), parameter L2 (lower)
usage: python eecc.py <results_dir> <out_prefix>"""
import glob, json, sys, collections
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*_*_*.json")) if "progress" not in f]
METH = [("ga_stop0", "GA만"), ("ga_repair0", "GA+보정"), ("sisa_after", "SISA S=5")]
def get(r, kind, key):
    s = r["scen"][kind]; return s[key] if not key.endswith("0") else s[key[:-1]][0]
def sape(u, s): return 0.0 if abs(u) + abs(s) == 0 else abs(u - s) / (abs(u) + abs(s)) * 100
def row(r, kind, key):
    m, ret = get(r, kind, key), r["scen"][kind]["retrain"]
    # time: GA/repair 'seconds' is GA(+repair) time; retrain seconds; SISA after = affected shards
    out = {"eff_time": ret["seconds"] / max(m["seconds"], 1e-9), "eff_pass": ret["cost_sample_passes"] / m["cost_sample_passes"],
           "effectiveness": abs(m["test_acc"] - ret["test_acc"]), "certif": sape(m["forget_acc"], ret["forget_acc"])}
    for k in ("cons_agree", "cons_kl", "cons_l2"):
        if k in m and m[k] is not None: out[k] = m[k]
    return out
tab = collections.defaultdict(lambda: collections.defaultdict(list))   # (kind, ds, model, meth) -> metric -> values
for r in recs:
    for kind in ("cls", "rand"):
        for key, _ in METH + [("original", "원본")]:
            if key == "original":
                m, ret = r["scen"][kind]["original"], r["scen"][kind]["retrain"]
                o = {"effectiveness": abs(m["test_acc"] - ret["test_acc"]), "certif": sape(m["forget_acc"], ret["forget_acc"])}
                for k in ("cons_agree", "cons_kl", "cons_l2"):
                    if k in m and m[k] is not None: o[k] = m[k]
            else: o = row(r, kind, key)
            for k, v in o.items(): tab[(kind, r["dataset"], r["model"], key)][k].append(v)
names = dict(METH + [("original", "원본")])
lines = []
mean = lambda a: float(np.mean(a))
for kind, ttl in (("cls", "클래스 3 잊기"), ("rand", "랜덤 1000개 잊기")):
    lines.append(f"\n## {ttl} - EECC (모델 4종·데이터셋 4종 평균; 값은 재학습 대비)")
    lines.append("| 모델 | 방법 | Efficiency(시간 배) | Efficiency(샘플 배) | Effectiveness(↓) | Certifiability SAPE%(↓) | Consistency 일치율(↑) | Consistency KL(↓) | Consistency L2(↓) |\n|---|---|---|---|---|---|---|---|---|")
    for model in ("cnn", "mlp", "vit", "resnet18"):
        for key in ("original", "ga_stop0", "ga_repair0", "sisa_after"):
            vs = collections.defaultdict(list)
            for (kd, ds, md, ky), mm in tab.items():
                if kd == kind and md == model and ky == key:
                    for k, v in mm.items(): vs[k].append(mean(v))
            if not vs: continue
            f = lambda k, fmt="%.3f": (fmt % mean(vs[k])) if vs.get(k) else "-"
            lines.append(f"| {model} | {names[key]} | {f('eff_time','%.1f')} | {f('eff_pass','%.1f')} | {f('effectiveness')} | {f('certif','%.1f')} | {f('cons_agree')} | {f('cons_kl')} | {f('cons_l2','%.2f')} |")
open(f"{OUT}.md", "w").write("\n".join(lines)); print("\n".join(lines))
json.dump({"|".join(k): {m: [float(x) for x in v] for m, v in mm.items()} for k, mm in tab.items()}, open(f"{OUT}.json", "w"))
