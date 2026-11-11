"""30라운드 스윕 그래프: (1) 학습률 비교  (2) lr=0.005에서 μ 비교.  결과: results/sweep30/*.json → report/sweep30_*.png"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
RES = HERE / "results" / "sweep30"
OUT = HERE / "report"
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("AppleGothic", "Malgun Gothic") if f in installed][:1] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(tag):
    e = json.loads((RES / f"{tag}.json").read_text())["server_eval"]
    xs = sorted(int(k) for k in e)
    return xs, [e[str(x)]["accuracy"] * 100 for x in xs], [e[str(x)]["loss"] for x in xs]


def tail(acc):
    return sum(acc[-5:]) / 5


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel("라운드 (0 = 초기 모델)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1) 학습률 비교: 정확도 곡선 + 마지막 5라운드 평균 막대
LRS = ("0.005", "0.01", "0.02")
LR_COL = {"0.005": "#2a78d6", "0.01": "#1f9e6a", "0.02": "#eb6834"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150, gridspec_kw={"width_ratios": [1.7, 1]})
for lr in LRS:
    for strat, ls in (("fedavg", "--"), ("fedprox_mu0.01", "-")):
        xs, acc, _ = load(f"r30_lr{lr}_{strat}")
        a1.plot(xs, acc, ls, color=LR_COL[lr], lw=1.8,
                label=f"lr={lr} · {'FedAvg' if strat == 'fedavg' else 'FedProx μ=0.01'}")
style(a1, "서버 정확도 (점선 FedAvg / 실선 FedProx μ=0.01)", "정확도 (%)")
a1.set_ylim(5, 62)
a1.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
w = 0.36
for i, lr in enumerate(LRS):
    for j, (strat, name) in enumerate((("fedavg", "FedAvg"), ("fedprox_mu0.01", "FedProx μ=0.01"))):
        v = tail(load(f"r30_lr{lr}_{strat}")[1])
        a2.bar(i + (j - 0.5) * w, v, w, color=LR_COL[lr], alpha=0.45 if j == 0 else 1.0,
               label=name if i == 0 else None, edgecolor="#333" if j else "none")
        a2.text(i + (j - 0.5) * w, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
a2.set_xticks(range(3))
a2.set_xticklabels([f"lr={lr}" for lr in LRS])
a2.set_ylim(30, 62)
style(a2, "마지막 5라운드(R26–30) 평균", "정확도 (%)")
a2.set_xlabel("")
a2.legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("학습률 비교 · Non-IID (Dirichlet α=0.3) · 30라운드", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "sweep30_lr.png", bbox_inches="tight")
plt.close(fig)

# 2) μ 비교 (lr=0.005)
MUS = [("FedAvg (μ=0)", "r30_lr0.005_fedavg", "#555555"),
       ("μ=0.001", "r30_lr0.005_fedprox_mu0.001", "#9ecae1"),
       ("μ=0.005", "r30_lr0.005_fedprox_mu0.005", "#6baed6"),
       ("μ=0.01", "r30_lr0.005_fedprox_mu0.01", "#2a78d6"),
       ("μ=0.02", "r30_lr0.005_fedprox_mu0.02", "#1f9e6a"),
       ("μ=0.05", "r30_lr0.005_fedprox_mu0.05", "#eb6834")]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
vals = []
for name, tag, c in MUS:
    xs, acc, loss = load(tag)
    a1.plot(xs, acc, "-", color=c, lw=1.8, label=name)
    a2.plot(xs, loss, "-", color=c, lw=1.8, label=name)
    vals.append((name, tail(acc), c))
style(a1, "서버 정확도", "정확도 (%)")
a1.set_ylim(5, 62)
style(a2, "서버 손실", "손실")
for i, (name, v, c) in enumerate(vals):
    a3.bar(i, v, 0.65, color=c)
    a3.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
a3.set_xticks(range(len(vals)))
a3.set_xticklabels(["0\n(FedAvg)", "0.001", "0.005", "0.01", "0.02", "0.05"], fontsize=8)
a3.set_ylim(45, 60)
style(a3, "R26–30 평균 정확도", "정확도 (%)")
a3.set_xlabel("μ")
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
fig.suptitle("FedProx μ 비교 (lr=0.005) · Non-IID (Dirichlet α=0.3) · 30라운드", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "sweep30_mu.png", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "sweep30_lr.png", OUT / "sweep30_mu.png")
