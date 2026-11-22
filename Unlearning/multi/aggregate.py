import glob, json, collections
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
D = "results"; recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*_*_*.json")) if "progress" not in f]
DS = ["mnist", "svhn", "cifar10", "cifar100"]; MD = ["cnn", "mlp", "vit", "resnet18"]
METH = [("original", "원본", "#888888"), ("retrain", "재학습", "#2a9d8f"), ("ga_stop0", "GA만", "#e76f51"), ("ga_repair0", "GA+보정", "#f4a261"), ("sisa_after", "SISA S=5", "#6a4c93")]
def get(r, kind, key):
    s = r["scen"][kind]
    return s[key] if not key.endswith("0") else s[key[:-1]][0]
agg = collections.defaultdict(list)
for r in recs: agg[(r["dataset"], r["model"])].append(r)
def stat(rs, kind, key, f):
    v = [f(get(r, kind, key)) for r in rs]; return float(np.mean(v)), float(np.std(v)), len(v)
# tables
lines = []
for kind, ttl in (("cls", "클래스 3 잊기"), ("rand", "랜덤 1000개 잊기")):
    for met, mn in (("forget_acc", "잊을 데이터 정확도"), ("test_acc", "테스트 정확도")):
        lines.append(f"\n## {ttl} - {mn}\n| 데이터셋/모델 | " + " | ".join(n for _, n, _ in METH) + " | n시드 |\n|---|" + "---|" * (len(METH) + 1))
        for d in DS:
            for m in MD:
                rs = agg.get((d, m))
                if not rs: continue
                cells = [f"{stat(rs, kind, k, lambda s: s[met])[0]:.3f}" for k, _, _ in METH]
                lines.append(f"| {d}/{m} | " + " | ".join(cells) + f" | {len(rs)} |")
open("summary.md", "w").write("\n".join(lines)); print("\n".join(lines))
# figure: heatmap-like grouped bars, one panel per (kind, metric)
for kind, ttl in (("cls", "클래스 3 잊기"), ("rand", "랜덤 1000개 잊기")):
    fig, ax = plt.subplots(2, 1, figsize=(17, 9))
    for a, (met, mn) in zip(ax, (("forget_acc", "잊을 데이터 정확도 (재학습 값에 가까울수록 좋음)"), ("test_acc", "테스트 정확도 (높을수록 좋음)"))):
        combos = [(d, m) for d in DS for m in MD if (d, m) in agg]; xs = np.arange(len(combos)); w = 0.16
        for j, (k, n, c) in enumerate(METH):
            vals = [stat(agg[cm], kind, k, lambda s: s[met])[0] for cm in combos]; errs = [stat(agg[cm], kind, k, lambda s: s[met])[1] for cm in combos]
            a.bar(xs + (j - 2) * w, vals, w, yerr=errs, color=c, label=n, capsize=1.5)
        a.set_xticks(xs); a.set_xticklabels([f"{d}\n{m}" for d, m in combos], fontsize=7); a.set_ylim(0, 1.05); a.set_title(f"{ttl}: {mn}", fontsize=10); a.legend(ncol=5, fontsize=8)
    plt.tight_layout(); plt.savefig(f"multi_{kind}.png", dpi=130); plt.close()
