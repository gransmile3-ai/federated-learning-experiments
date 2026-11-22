import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
R = json.load(open("results_svhn_ga/svhn_ga.json"))
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
for j, (kind, ttl) in enumerate((("rand", "랜덤 샘플 1000개 잊기"), ("cls", "숫자 3 클래스 통째로 잊기"))):
    a = ax[0][j]
    for lr, c in zip((1e-3, 3e-3, 1e-2), ("#2a9d8f", "#e9c46a", "#e76f51")):
        tr = [r["scen"][kind]["sweep"][[1e-3, 3e-3, 1e-2].index(lr)]["traj"] for r in R][0]  # seed 0
        st = [t["step"] for t in tr]
        a.plot(st, [t["forget_acc"] for t in tr], c=c, ls="-", label=f"잊을 데이터 acc, lr={lr:g}")
        a.plot(st, [t["test_acc"] for t in tr], c=c, ls="--", label=f"테스트 acc, lr={lr:g}")
    a.axhline(R[0]["scen"][kind]["retrain"]["forget_acc"], c="k", ls=":", lw=1, label="재학습 forget acc")
    a.set_title(f"{ttl}: GA 진행 (시드 0, 실선=잊을 데이터, 점선=테스트)", fontsize=10); a.set_xlabel("GA 스텝"); a.set_ylim(0, 1); a.legend(fontsize=6.5, ncol=2)
    b = ax[1][j]; names = ["원본", "재학습", "GA 멈춤\nlr1e-3", "GA+보정\nlr1e-3", "GA 멈춤\nlr3e-3", "GA+보정\nlr3e-3"]
    def get(nm, i=None): return [r["scen"][kind][nm] if i is None else r["scen"][kind][nm][i] for r in R]
    sets = [get("original"), get("retrain"), get("stop", 0), get("stop_repair", 0), get("stop", 1), get("stop_repair", 1)]
    xs = np.arange(len(sets)); w = .38
    for off, key, col, lab in ((-w / 2, "forget_acc", "#e76f51", "잊을 데이터 acc"), (w / 2, "test_acc", "#264653", "테스트 acc")):
        m = [np.mean([s[key] for s in S]) for S in sets]; e = [np.std([s[key] for s in S]) for S in sets]
        b.bar(xs + off, m, w, yerr=e, color=col, capsize=3, label=lab)
        for x_, v in zip(xs + off, m): b.text(x_, v + .02, f"{v:.2f}", ha="center", fontsize=7)
    b.set_xticks(xs); b.set_xticklabels(names, fontsize=8); b.set_ylim(0, 1.08); b.legend(fontsize=8); b.set_title(f"{ttl}: 최종 비교 (3시드)", fontsize=10)
plt.tight_layout(); plt.savefig("svhn_ga.png", dpi=140)
