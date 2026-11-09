"""FedAvg(quickstart-pytorch/aws_run.txt) vs FedProx(FedProx/aws_run.txt) 비교 그래프."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).parent))
from plot_run import parse  # noqa: E402

installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("AppleGothic", "Malgun Gothic") if f in installed][:1] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

runs = {
    "FedAvg": Path(__file__).parent.parent / "quickstart-pytorch" / "aws_run.txt",
    "FedProx (μ=0.1)": Path(__file__).parent / "aws_run.txt",
}
colors = {"FedAvg": "#2a78d6", "FedProx (μ=0.1)": "#eb6834"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
for name, path in runs.items():
    server, _, _ = parse(path)
    xs = sorted(server)
    a1.plot(xs, [server[r]["accuracy"] * 100 for r in xs], "o-", color=colors[name], label=name, lw=2)
    a2.plot(xs, [server[r]["loss"] for r in xs], "o-", color=colors[name], label=name, lw=2)
    a1.annotate(f"{server[xs[-1]]['accuracy']*100:.2f}%", (xs[-1], server[xs[-1]]["accuracy"] * 100),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=colors[name])
for ax, t, y in ((a1, "서버 전역 평가 정확도", "accuracy (%)"), (a2, "서버 전역 평가 손실", "loss")):
    ax.set_title(t, loc="left", fontweight="bold")
    ax.set_xlabel("라운드 (0 = 초기 모델)")
    ax.set_ylabel(y)
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("FedAvg vs FedProx · AWS SuperLink + Mac/Desktop SuperNode · CIFAR-10 IID 2분할", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
out = Path(__file__).parent / "compare_fedavg_fedprox.png"
fig.savefig(out)
print("saved:", out)
