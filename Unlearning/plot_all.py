import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
R = json.load(open("results/results.json")); S = R["stage2"]; SI = json.load(open("results_sisa/sisa_results.json"))
ln10 = math.log(10)
g = lambda f: next(x for x in S if f(x))
ga = g(lambda x: x["kind"] == "ga_stop" and abs(x["target"] - ln10) < 1e-6)
gar = g(lambda x: x["kind"] == "ga_repair" and abs(x["target"] - ln10) < 1e-6 and x["repair_rounds"] == 1)
fe = g(lambda x: x["kind"] == "federaser_mix" and x["alpha"] == 0.25)
def sisa(k): return np.mean([s["drop_desktop"][k] for s in SI]), np.std([s["drop_desktop"][k] for s in SI])
sk = {"global_test_acc": "test_acc", "forget_train_acc": "desktop_train_acc", "mia_auc": "mia_client"}
rows = [("원본 FedAvg", R["original"], "#888888"), ("재학습 (기준선)", R["retrain"], "#2a9d8f"),
        ("업데이트 빼기", R["subtract"], "#cccccc"), ("GA만", ga, "#e76f51"),
        ("GA+보정 1라운드", gar, "#f4a261"), ("FedEraser 변형", fe, "#264653"), ("SISA (샤드 제거)", None, "#6a4c93")]
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
titles = {"global_test_acc": "테스트 정확도 (높을수록 유지)", "forget_train_acc": "잊을 데이터(데스크탑) 정확도\n(재학습 값에 가까울수록 좋음)", "mia_auc": "MIA AUC (0.5 = 멤버 구분 불가)"}
for a, k in zip(ax, titles):
    vals, errs = [], []
    for n, r, c in rows:
        if r is None: m, e = sisa(sk[k])
        else: m, e = r[k], 0
        vals.append(m); errs.append(e)
    names = [n for n, _, _ in rows]
    a.barh(names, vals, xerr=errs, color=[c for _, _, c in rows], capsize=3); a.invert_yaxis(); a.set_title(titles[k], fontsize=10)
    for i, (v, e) in enumerate(zip(vals, errs)): a.text(v + e + .005, i, f"{v:.3f}", va="center", fontsize=8)
    ref = R["retrain"][k]; a.axvline(ref, ls=":", c="#2a9d8f", lw=1)
    if k == "mia_auc": a.axvline(0.5, ls="--", c="k", lw=.6)
    if k != "global_test_acc": a.set_yticklabels([])
fig.suptitle("데스크탑 클라이언트 언러닝 방법 비교 (점선 = 재학습 기준선 값, SISA는 3시드 평균±표준편차)", fontsize=11)
plt.tight_layout(); plt.savefig("compare_all.png", dpi=140)
# trade-off: cost vs distance to retrain on forget acc
fig, ax = plt.subplots(figsize=(8.5, 5.5))
pts = [("GA만", ga["local_epoch_cost"], ga, "#e76f51"), ("GA+보정 1라운드", gar["local_epoch_cost"], gar, "#f4a261"),
       ("FedEraser 변형", fe["local_epoch_cost"], fe, "#264653"), ("업데이트 빼기", 0, R["subtract"], "#999999"),
       ("재학습", R["retrain"]["local_epoch_cost"], R["retrain"], "#2a9d8f")]
ref = R["retrain"]["forget_train_acc"]
for n, cst, r, c in pts:
    ax.scatter(max(cst, 0.1), abs(r["forget_train_acc"] - ref), s=r["global_test_acc"] * 1200, c=c, alpha=.8)
    ax.annotate(f'{n}\n(테스트 {r["global_test_acc"]:.2f})', (max(cst, 0.1), abs(r["forget_train_acc"] - ref)), textcoords="offset points", xytext=(10, 8), fontsize=8)
sm, _ = sisa("desktop_train_acc"); st, _ = sisa("test_acc")
ax.scatter(0.1, abs(sm - ref), s=st * 1200, c="#6a4c93", alpha=.8)
ax.annotate(f"SISA 샤드 제거\n(테스트 {st:.2f}, 비용 0)", (0.1, abs(sm - ref)), textcoords="offset points", xytext=(10, 8), fontsize=8)
ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0", "1", "10", "100"]); ax.minorticks_off(); ax.set_xlabel("언러닝 비용 (local epoch, 로그축, 0은 0.1 위치)"); ax.set_ylabel("재학습과의 차이: 잊을 데이터 정확도 |Δ|")
ax.set_title("비용 vs 잊은 정도 (왼쪽 아래일수록 좋음, 원 크기 = 테스트 정확도)"); ax.set_xlim(0.05, 300); ax.set_ylim(-0.03, 0.6); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("tradeoff.png", dpi=140)
