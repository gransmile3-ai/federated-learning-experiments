#!/usr/bin/env python3
"""학습률 + μ 스윕 (30라운드). Oracle 서버 혼자서 SuperLink + SuperNode 2개로 실행.

1단계: lr {0.005, 0.01, 0.02} × {FedAvg, FedProx μ=0.01}
2단계: 1단계에서 가장 좋은 lr(두 전략의 R26–R30 평균 정확도 평균)로 μ {0.001, 0.005, 0.02, 0.05}
결과: ~/NonIID/sweep30/<tag>.json, 진행 상황: ~/NonIID/sweep30/progress.txt
"""
import json, statistics, subprocess, time
from pathlib import Path

SRC = Path.home() / "fedcompare_src"
OUT = Path.home() / "NonIID"
LOG = OUT / "sweep30"
LOG.mkdir(parents=True, exist_ok=True)
FLWR = str(Path.home() / "fl/.venv/bin/flwr")
ROUNDS = 30
PROG = LOG / "progress.txt"


def note(msg):
    with open(PROG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M ") + msg + "\n")


def tag_of(strategy, lr, mu):
    return f"r30_lr{lr}_" + ("fedavg" if strategy == "fedavg" else f"fedprox_mu{mu}")


def tail_mean(tag):
    d = json.loads((OUT / f"{tag}.json").read_text())
    e = d["server_eval"]
    acc = [e[str(r)]["accuracy"] * 100 for r in range(ROUNDS - 4, ROUNDS + 1)]
    best = max(v["accuracy"] for v in e.values()) * 100
    return statistics.mean(acc), best, e[str(ROUNDS)]["accuracy"] * 100


def run(strategy, lr, mu=0.0):
    tag = tag_of(strategy, lr, mu)
    if (OUT / f"{tag}.json").exists():
        note(f"skip {tag} (이미 있음)")
        return tag
    cfg = f'strategy="{strategy}" learning-rate={lr} num-server-rounds={ROUNDS} run-tag="{tag}"'
    if strategy == "fedprox":
        cfg += f" proximal-mu={mu}"
    note(f"시작 {tag}")
    t0 = time.time()
    with open(LOG / f"{tag}.txt", "w") as f:
        rc = subprocess.run([FLWR, "run", ".", "local-oci", "--stream", "--run-config", cfg],
                            cwd=SRC, stdout=f, stderr=subprocess.STDOUT).returncode
    errs = (LOG / f"{tag}.txt").read_text(errors="replace").count("error in reply")
    mins = (time.time() - t0) / 60
    if (OUT / f"{tag}.json").exists():
        m, b, fin = tail_mean(tag)
        note(f"완료 {tag}: 최종 {fin:.2f}% · 최고 {b:.2f}% · R26-30 평균 {m:.2f}% ({mins:.0f}분, rc={rc}, 오류 {errs})")
    else:
        note(f"실패 {tag}: 결과 파일 없음 (rc={rc}, 오류 {errs}, {mins:.0f}분)")
    return tag


note("=== 스윕 시작: 30라운드, 1단계 lr 비교")
scores = {}
for lr in (0.005, 0.01, 0.02):
    vals = []
    for strategy, mu in (("fedavg", 0.0), ("fedprox", 0.01)):
        tag = run(strategy, lr, mu)
        if (OUT / f"{tag}.json").exists():
            vals.append(tail_mean(tag)[0])
    if len(vals) == 2:
        scores[lr] = statistics.mean(vals)
best_lr = max(scores, key=scores.get)
note(f"=== 1단계 끝: lr별 점수 {', '.join(f'{k}={v:.2f}%' for k, v in scores.items())} → 2단계 lr={best_lr}")
for mu in (0.001, 0.005, 0.02, 0.05):
    run("fedprox", best_lr, mu)
note("=== 스윕 전체 완료")
(LOG / "DONE").write_text(str(best_lr))
