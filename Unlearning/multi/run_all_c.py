import os, subprocess, sys, time
PY = os.path.expanduser("~/FL/.venv-1/bin/python"); D = os.path.dirname(os.path.abspath(__file__))
dsets = ["mnist", "svhn", "cifar10", "cifar100"]
plan = []
plan = []
for seed in (0, 1, 2):
    for m in ("cnn", "mlp"):
        plan += [(d, m, seed) for d in dsets]
for d, m, s in plan:
    out = f"{D}/results_c/{d}_{m}_{s}.json"
    if os.path.exists(out): continue
    t = time.time()
    r = subprocess.run([PY, f"{D}/multi_run_c.py", d, m, str(s), out], capture_output=True, text=True, cwd=D)
    with open(f"{D}/results_c/progress.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {d}/{m}/s{s} rc={r.returncode} {time.time()-t:.0f}s\n")
        f.write("\n".join(l for l in r.stdout.splitlines() if l.startswith("[")) + "\n")
        if r.returncode: f.write(r.stderr[-1500:] + "\n")
open(f"{D}/results_c/DONE", "w").write("done")
