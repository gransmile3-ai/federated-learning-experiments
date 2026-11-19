"""Stage 2: stabilised unlearning variants, using models/history saved by unlearn.py."""
import json, sys, time, math
import torch, torch.nn.functional as F
import unlearn as U
from unlearn import Net, flat, setflat, evaluate, metrics, local_train, FORGET, KEEP

OUT = "/Users/yjj/FL/Unlearning/results"
data = U.load_all()
res = json.load(open(f"{OUT}/results.json"))
w_orig = torch.load(f"{OUT}/original.pt")["w"]
hist = torch.load(f"{OUT}/orig_history.pt")["hist"]
torch.manual_seed(42); init = flat(Net())
bpe = len(data[FORGET]["train"][1]) / U.BS
sub_idx = torch.randperm(len(data[FORGET]["train"][1]), generator=torch.Generator().manual_seed(3))[:2048]
forget_sub = (data[FORGET]["train"][0][sub_idx], data[FORGET]["train"][1][sub_idx])


def ga_stop(w, lr=1e-3, target=math.log(10), max_steps=300, check=5):
    """Gradient ascent run by the desktop client; stop once its own loss reaches `target`
    (= loss of a uniform guess, ln 10) -- no access to the retrained oracle."""
    net = Net(); setflat(net, w.clone()); net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    dl = U.loader(data[FORGET]["train"], True, seed=7); it = iter(dl); steps = 0
    while steps < max_steps:
        try: x, y = next(it)
        except StopIteration: it = iter(dl); x, y = next(it)
        opt.zero_grad(); (-F.cross_entropy(net(x), y)).backward(); opt.step(); steps += 1
        if steps % check == 0 and evaluate(net, forget_sub)["loss"] >= target:
            break
    return flat(net), steps


def repair(w, rounds):
    """Remaining client (Mac) alone: FedAvg rounds starting from w."""
    for r in range(rounds):
        m = Net(); setflat(m, w.clone()); local_train(m, data[KEEP]["train"], U.EPOCHS, seed=9000 + r); w = flat(m)
    return w


res["stage2"] = []
for target in (math.log(10), 3.0):
    t0 = time.time(); w_ga, steps = ga_stop(w_orig, target=target)
    secs = time.time() - t0
    r = metrics(w_ga, data, f"ga_stop(loss>={target:.2f}) steps={steps}", steps / bpe, secs)
    r.update(kind="ga_stop", target=target, steps=steps); res["stage2"].append(r)
    for k in (1, 2, 3, 5):
        t1 = time.time(); w_r = repair(w_ga, k)
        r = metrics(w_r, data, f"ga_stop(loss>={target:.2f})+repair{k}", steps / bpe + k * U.EPOCHS, secs + time.time() - t1)
        r.update(kind="ga_repair", target=target, steps=steps, repair_rounds=k); res["stage2"].append(r)


def federaser_mix(alpha, dt, e_cal):
    """FedEraser with the old direction blended with the freshly computed direction:
    cal = |new| * normalise(alpha*old_dir + (1-alpha)*new_dir) per layer (alpha=1 is vanilla FedEraser)."""
    net = Net(); sl = U.layer_slices(net); w = init.clone(); used = 0
    for s in range(0, len(hist), dt):
        old = sum(hist[r]["updates"][KEEP] for r in range(s, min(s + dt, len(hist))))
        m = Net(); setflat(m, w.clone()); local_train(m, data[KEEP]["train"], e_cal, seed=5000 + s); used += e_cal
        new = flat(m) - w; cal = torch.zeros_like(w)
        for a, b in sl:
            d = alpha * old[a:b] / (old[a:b].norm() + 1e-12) + (1 - alpha) * new[a:b] / (new[a:b].norm() + 1e-12)
            cal[a:b] = new[a:b].norm() * d / (d.norm() + 1e-12)
        w = w + cal
    return w, used


for alpha, dt, e in [(0.5, 1, 1), (0.25, 1, 1), (0.5, 2, 2), (0.0, 2, 2)]:
    t0 = time.time(); w, used = federaser_mix(alpha, dt, e)
    r = metrics(w, data, f"federaser alpha={alpha} dt={dt} e_cal={e}", used, time.time() - t0)
    r.update(kind="federaser_mix", alpha=alpha, dt=dt, e_cal=e); res["stage2"].append(r)

json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
print("saved stage 2")
