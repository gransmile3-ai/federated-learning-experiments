import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
G = json.load(open("results_svhn_ga/svhn_ga.json")); S = json.load(open("results_svhn_ga/svhn_sisa_metrics.json"))
G = [g for g in G if g["seed"] in {s["seed"] for s in S}]
names = ["원본", "재학습\n(단일)", "GA만\nlr1e-3", "GA+보정\nlr1e-3", "GA+보정\nlr3e-3", "SISA\nS=5", "SISA\nS=10"]
cols = ["#888888", "#2a9d8f", "#e76f51", "#f4a261", "#e9c46a", "#6a4c93", "#9d7fd0"]
def rows(kind):
    out = []
    for i in range(len(G)):
        sc = G[i]["scen"][kind]; sd = S[i]["S"]
        out.append([sc["original"], sc["retrain"], sc["stop"][0], sc["stop_repair"][0], sc["stop_repair"][1], sd["5"][f"{kind}_after"], sd["10"][f"{kind}_after"]])
    return out  # [seed][method]
def stat(R, f):
    v = np.array([[f(m, R[s][0] if True else None) for m in R[s]] for s in range(len(R))], dtype=float)
    return v.mean(0), v.std(0)
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
summary = []
for r, (kind, ttl) in enumerate((("rand", "랜덤 샘플 1000개 잊기"), ("cls", "숫자 3 클래스 잊기"))):
    R = rows(kind); base = [row[1]["cost_sample_passes"] for row in R]
    defs = [("잊을 데이터 정확도 (재학습에 가까울수록 좋음)", lambda m, s, b=None: m["forget_acc"]),
            ("테스트 정확도 (높을수록 좋음)", lambda m, s, b=None: m["test_acc"]),
            ("비용: 재학습 대비 학습량 (로그, 낮을수록 좋음)", None),
            ("MIA AUC (0.5에 가까울수록)" if kind == "rand" else "숫자 3 테스트 정확도 (0에 가까울수록 잊음)", lambda m, s, b=None: m["mia_auc"] if kind == "rand" else m["test_forgetclass_acc"])]
    for c, (t, f) in enumerate(defs):
        a = ax[r][c]
        if f is None:
            v = np.array([[m["cost_sample_passes"] / base[s] for m in R[s]] for s in range(len(R))])
        else:
            v = np.array([[f(m, None) for m in R[s]] for s in range(len(R))])
        mean, sd = v.mean(0), v.std(0)
        idx = list(range(len(names)))
        if f is None: idx = idx[1:]
        a.bar([names[i] for i in idx], [mean[i] for i in idx], yerr=[sd[i] for i in idx], color=[cols[i] for i in idx], capsize=3)
        for k, i in enumerate(idx): a.text(k, mean[i] * (1.15 if f is None else 1) + (0 if f is None else 0.02), f"{mean[i]:.3f}" if f else f"{mean[i]:.3f}x", ha="center", fontsize=7)
        if f is None: a.set_yscale("log"); a.set_ylim(0.003, 3); a.set_yticks([0.01, 0.1, 1]); a.set_yticklabels(["0.01x", "0.1x", "1x"]); a.minorticks_off()
        else: a.set_ylim(0, 1.08 if not (kind == "rand" and c == 3) else 0.75)
        if c == 0 or (c == 1): a.axhline(mean[1], color="#2a9d8f", ls=":", lw=1)
        if kind == "rand" and c == 3: a.axhline(0.5, color="k", ls="--", lw=.7)
        a.set_title(f"{ttl}\n{t}", fontsize=9); a.tick_params(axis="x", labelsize=7)
        summary.append((kind, t, mean.round(3).tolist(), sd.round(3).tolist()))
fig.suptitle("SVHN: Gradient Ascent vs SISA (같은 잊을 대상, 3시드 평균±표준편차, 점선=재학습 값)", fontsize=12)
plt.tight_layout(); plt.savefig("svhn_ga_vs_sisa.png", dpi=140)
print("methods", [n.replace("\n", " ") for n in names])
for s in summary: print(s[0], "|", s[1][:28], "|", s[2])
