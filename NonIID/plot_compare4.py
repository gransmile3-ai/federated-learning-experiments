"""Non-IID(Dirichlet α=0.3) FedAvg vs FedProx μ=0.01/0.1/1.0 비교 그래프."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
OUT = HERE / "report"
OUT.mkdir(exist_ok=True)
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("AppleGothic", "Malgun Gothic") if f in installed][:1] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

runs = [("FedAvg", "noniid_fedavg", "#2a78d6"),
        ("FedProx μ=0.01", "noniid_fedprox_mu0.01", "#1f9e6a"),
        ("FedProx μ=0.1", "noniid_fedprox_mu0.1", "#eb6834"),
        ("FedProx μ=1.0", "noniid_fedprox_mu1", "#9b59b6")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
for name, tag, c in runs:
    e = json.loads((HERE / f"results/{tag}.json").read_text())["server_eval"]
    xs = sorted(int(k) for k in e)
    acc = [e[str(x)]["accuracy"] * 100 for x in xs]
    loss = [e[str(x)]["loss"] for x in xs]
    a1.plot(xs, acc, "o-", color=c, lw=2, ms=4, label=f"{name} ({acc[-1]:.1f}%)")
    a2.plot(xs, loss, "o-", color=c, lw=2, ms=4, label=name)
for ax, t, y in ((a1, "서버 평가 정확도", "정확도 (%)"), (a2, "서버 평가 손실", "손실")):
    ax.set_title(t, loc="left", fontweight="bold")
    ax.set_xlabel("라운드")
    ax.set_ylabel(y)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Non-IID (Dirichlet α=0.3, 클라이언트 2개, 10라운드) — FedAvg vs FedProx μ 비교", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "noniid_compare_mu.png")
print(OUT / "noniid_compare_mu.png")
