s = open("multi_run.py").read()
# 1) Eval: keep reference (retrain) and compute consistency vs it
s = s.replace("""        self.yf, self.yr, self.yt = y[fidx], y[reval], yt; self.hidx = hidx
""", """        self.yf, self.yr, self.yt = y[fidx], y[reval], yt; self.hidx = hidx
        self.ref = None
""")
s = s.replace("""        if self.kind == "rand": r["mia_auc"] =""", """        if self.ref is not None:
            rp, rw = self.ref
            r["cons_agree"] = (pt.argmax(1) == rp.argmax(1)).float().mean().item()          # Consistency_y (fraction, test set)
            r["cons_kl"] = (rp * (torch.log(rp.clamp_min(1e-12)) - torch.log(pt.clamp_min(1e-12)))).sum(1).mean().item()  # KL(retrain || unlearned)
            r["cons_l2"] = float((flatw(states[0]) - rw).norm()) if len(states) == 1 else None  # Consistency_theta (single models only)
        self.last_pt = pt
        if self.kind == "rand": r["mia_auc"] =""")
s = s.replace("def sd(net):", """def flatw(st):
    return torch.cat([v.detach().float().flatten() for k, v in st.items() if v.is_floating_point()])


def sd(net):""")
# 2) after retrain: set reference and recompute original with consistency
s = s.replace("""        S_["retrain"] = ev.metrics(model, nc, [sd(net)], "retrain", sp, time.time() - t0)
""", """        S_["retrain"] = ev.metrics(model, nc, [sd(net)], "retrain", sp, time.time() - t0)
        ev.ref = (ev.last_pt, flatw(sd(net)))
        S_["original"] = ev.metrics(model, nc, [w_orig], "original", T_EPOCHS * N, rec["orig_seconds"])
""")
open("multi_run_c.py", "w").write(s)
r = open("run_all.py").read()
r = r.replace("multi_run.py", "multi_run_c.py").replace("results/", "results_c/").replace("/results\"", "/results_c\"")
r = r.replace('for seed in (0,):\n    for m in ("cnn", "mlp", "vit"):\n        plan += [(d, m, seed) for d in dsets]\nplan += [(d, "resnet18", 0) for d in dsets]\nfor seed in (1, 2):\n    for m in ("cnn", "mlp", "vit"):', 'plan = []\nfor seed in (0, 1, 2):\n    for m in ("cnn", "mlp"):')
open("run_all_c.py", "w").write(r)
