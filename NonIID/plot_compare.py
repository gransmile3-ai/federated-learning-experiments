"""Non-IID(Dirichlet α=0.3) FedAvg vs FedProx(μ=0.01/0.1/1.0) 비교 그래프."""
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

runs = {
    "FedAvg": json.loads((HERE / "results/noniid_fedavg.json").read_text()),
    "FedProx (μ=0.01)": json.loads((HERE / "results/noniid_fedprox_mu0.01.json").read_text()),
    "FedProx (μ=0.1)": json.loads((HERE / "results/noniid_fedprox_mu0.1.json").read_text()),
    "FedProx (μ=1.0)": json.loads((HERE / "results/noniid_fedprox_mu1.json").read_text()),
}
colors = {"FedAvg": "#2a78d6", "FedProx (μ=0.01)": "#1f9e6a", "FedProx (μ=0.1)": "#eb6834", "FedProx (μ=1.0)": "#9b59b6"}


def series(d, key, metric):
    xs = sorted(int(k) for k in d[key])
    return xs, [d[key][str(x)][metric] for x in xs]


def style(ax, title, ylabel, xlabel="라운드"):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2 if "분포" in title else 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1) 4-panel 비교
fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150)
for name, d in runs.items():
    c = colors[name]
    xs, ys = series(d, "server_eval", "accuracy")
    ys = [y * 100 for y in ys]
    axs[0, 0].plot(xs, ys, "o-", color=c, lw=2, label=name)
    axs[0, 0].annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=9, color=c)
    xs, ys = series(d, "server_eval", "loss")
    axs[0, 1].plot(xs, ys, "o-", color=c, lw=2, label=name)
    best = min(range(len(ys)), key=ys.__getitem__)
    axs[0, 1].scatter([xs[best]], [ys[best]], s=120, facecolors="none", edgecolors=c, lw=1.5)
    xs, ys = series(d, "client_eval", "eval_acc")
    axs[1, 0].plot(xs, [y * 100 for y in ys], "o-", color=c, lw=2, label=name)
    xs, ys = series(d, "client_train", "train_loss")
    axs[1, 1].plot(xs, ys, "o-", color=c, lw=2, label=name)
style(axs[0, 0], "서버 전역 평가 정확도 (CIFAR-10 test 10,000장)", "accuracy (%)", "라운드 (0 = 초기 모델)")
style(axs[0, 1], "서버 전역 평가 손실 (○ = 최저점)", "loss", "라운드 (0 = 초기 모델)")
style(axs[1, 0], "클라이언트 로컬 평가 정확도 (가중 평균)", "accuracy (%)")
style(axs[1, 1], "클라이언트 학습 손실 (FedProx는 근접항 포함)", "train loss")
for ax in axs.flat:
    ax.set_xticks(range(0, 11))
axs[0, 0].set_ylim(5, 62)
axs[0, 0].set_xlim(-0.3, 11)
fig.suptitle("FedAvg vs FedProx (μ=0.01 / 0.1 / 1.0) · Non-IID (Dirichlet α=0.3) · Mac/Desktop SuperNode",
             x=0.01, ha="left", fontweight="bold")
h, l = axs[0, 0].get_legend_handles_labels()
for ax in axs.flat:
    ax.get_legend().remove()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT / "noniid_compare.png", bbox_inches="tight")
plt.close(fig)

# 2) 클라이언트별 라벨 분포 (label_hist 결과, seed=42)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
hist = {
    "Desktop (partition 0)": [1829, 4, 140, 3880, 922, 590, 4978, 4705, 4931, 505],
    "Mac (partition 1)": [3171, 4996, 4860, 1120, 4078, 4410, 22, 295, 69, 4495],
}
fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
w = 0.4
for i, (name, h) in enumerate(hist.items()):
    xs = [x + (i - 0.5) * w for x in range(10)]
    ax.bar(xs, h, w, label=f"{name} · {sum(h):,}장", color=["#7a5cc4", "#2fa37a"][i])
ax.set_xticks(range(10))
ax.set_xticklabels(classes)
ax.set_ylim(0, 6300)
style(ax, "클라이언트별 데이터 라벨 분포 (Dirichlet α=0.3, seed=42 · 80% 학습 / 20% 로컬 평가로 분할)", "이미지 수", "CIFAR-10 클래스")
fig.tight_layout()
fig.savefig(OUT / "label_distribution.png", bbox_inches="tight")
plt.close(fig)
print("saved", *sorted(p.name for p in OUT.glob("*.png")))
