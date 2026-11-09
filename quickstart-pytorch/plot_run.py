"""Flower 실행 로그(예: aws_run.txt)를 파싱해 라운드별 정확도/손실 그래프를 그린다.

사용법:
    python plot_run.py aws_run.txt            # aws_run.png 생성
    python plot_run.py aws_run.txt -o out.png
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt

ANSI = re.compile(r"\x1b\[[0-9;]*m|\[\d+m")
DICT = re.compile(r"(\{.*\})")

# 차트 팔레트 (라이트 모드)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_SERVER = "#2a78d6"  # 서버 측 전역 평가
C_CLIENT = "#eb6834"  # 클라이언트 평가(가중 평균)
C_TRAIN = "#1baf7a"   # 클라이언트 학습 손실


def parse(path):
    server, client_eval, train = {}, {}, {}
    rnd = None
    in_global = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = ANSI.sub("", raw)
        m_round = re.search(r"\[ROUND (\d+)/\d+\]", line)
        if m_round:
            rnd = int(m_round.group(1))
            continue
        m_dict = DICT.search(line)
        if "Initial global evaluation results" in line and m_dict:
            server[0] = ast.literal_eval(m_dict.group(1))
        elif "Global evaluation" in line:
            in_global = True
        elif "MetricRecord:" in line and m_dict and rnd is not None:
            rec = ast.literal_eval(m_dict.group(1))
            if in_global:
                server[rnd] = rec
                in_global = False
            elif "train_loss" in rec:
                train[rnd] = rec
            elif "eval_loss" in rec:
                client_eval[rnd] = rec
        elif "Final results" in line:
            break
    return server, client_eval, train


def style_axes(ax, title, ylabel, rounds):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=12)
    ax.set_xlabel("라운드 (0 = 초기 모델)", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xticks(rounds)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def line(ax, data, key, color, label, fmt):
    xs = sorted(data)
    ys = [data[r][key] for r in xs]
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    # 마지막 값만 직접 라벨 (위치는 place_end_labels에서 겹치지 않게 배치)
    ax._end_labels = getattr(ax, "_end_labels", []) + [(xs[-1], ys[-1], fmt.format(ys[-1]))]


def place_end_labels(ax, min_gap_pt=12):
    """끝점 라벨을 세로로 최소 간격만큼 벌려 겹치지 않게 그린다."""
    labels = getattr(ax, "_end_labels", [])
    if not labels:
        return
    min_gap_px = min_gap_pt * ax.figure.dpi / 72
    to_px = ax.transData.transform
    pts = sorted(((to_px((x, y))[1], x, y, t) for x, y, t in labels), key=lambda p: p[0])
    placed = []
    for py, x, y, t in pts:
        target = py if not placed else max(py, placed[-1] + min_gap_px)
        placed.append(target)
        dy_pt = (target - py) * 72 / ax.figure.dpi
        ax.annotate(t, (x, y), xytext=(7, dy_pt), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Flower 실행 로그 파일")
    p.add_argument("-o", "--out", help="출력 PNG 경로 (기본: <log>.png)")
    args = p.parse_args()

    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ("AppleGothic", "Malgun Gothic", "NanumGothic") if f in installed]
    plt.rcParams["font.family"] = korean[:1] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    server, client_eval, train = parse(args.log)
    rounds = sorted(set(server) | set(client_eval) | set(train))

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160,
                                         facecolor=SURFACE)

    # 정확도
    style_axes(ax_acc, "정확도", "accuracy", rounds)
    line(ax_acc, server, "accuracy", C_SERVER, "서버 전역 평가 (accuracy)", "{:.1%}")
    line(ax_acc, client_eval, "eval_acc", C_CLIENT, "클라이언트 평가 평균 (eval_acc)", "{:.1%}")
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_acc.set_ylim(0, None)
    ax_acc.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    # 손실
    style_axes(ax_loss, "손실", "loss", rounds)
    line(ax_loss, server, "loss", C_SERVER, "서버 전역 평가 (loss)", "{:.3f}")
    line(ax_loss, client_eval, "eval_loss", C_CLIENT, "클라이언트 평가 평균 (eval_loss)", "{:.3f}")
    line(ax_loss, train, "train_loss", C_TRAIN, "클라이언트 학습 평균 (train_loss)", "{:.3f}")
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")

    for ax in (ax_acc, ax_loss):
        ax.set_xlim(rounds[0] - 0.2, rounds[-1] + 0.5)

    fig.suptitle(f"FedAvg 실행 결과 · {Path(args.log).name}", x=0.01, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    fig.canvas.draw()  # 좌표 변환 확정 후 라벨 배치
    for ax in (ax_acc, ax_loss):
        place_end_labels(ax)

    out = args.out or str(Path(args.log).with_suffix(".png"))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
