import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
R = json.load(open("results_svhn/svhn_sisa.json")); Ss = ["5", "10", "20"]; Ks = ["1", "10", "100"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
nv = [r["naive"]["test_acc"] for r in R]
m = [np.mean([r["sisa"][S]["full"]["test_acc"] for r in R]) for S in Ss]; e = [np.std([r["sisa"][S]["full"]["test_acc"] for r in R]) for S in Ss]
ax[0].bar([f"S={S}" for S in Ss], m, yerr=e, color="#6a4c93", capsize=4)
ax[0].axhline(np.mean(nv), color="#2a9d8f", ls="--", label=f"단일 모델(9 epoch) {np.mean(nv):.3f}"); ax[0].legend()
for i, v in enumerate(m): ax[0].text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_ylim(0, 1); ax[0].set_title("SISA 앙상블 테스트 정확도 (SVHN, 3시드)\n샤드가 많을수록 하락 (S=20은 학습 부족 영향 큼)", fontsize=10)
w = 0.25; cols = ["#264653", "#e9c46a", "#e76f51"]
for j, k in enumerate(Ks):
    v = [np.mean([r["sisa"][S]["del"][k]["speedup_vs_naive"] for r in R]) for S in Ss]
    err = [np.std([r["sisa"][S]["del"][k]["speedup_vs_naive"] for r in R]) for S in Ss]
    ax[1].bar(np.arange(3) + (j - 1) * w, v, w, yerr=err, color=cols[j], capsize=3, label=f"삭제 {k}개")
    for i, x in enumerate(v): ax[1].text(i + (j - 1) * w, x * 1.08, f"{x:.1f}x", ha="center", fontsize=8)
ax[1].axhline(1, color="k", ls=":", lw=.8); ax[1].set_yscale("log"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"S={S}" for S in Ss])
ax[1].set_ylim(0.8, 60); ax[1].set_title("재학습 속도 향상 (처음부터 재학습 대비, 학습 샘플 수 기준)\n1x = 이득 없음", fontsize=10); ax[1].legend()
plt.tight_layout(); plt.savefig("svhn_sisa.png", dpi=140)
