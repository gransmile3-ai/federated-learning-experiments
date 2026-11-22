import json, numpy as np
R = json.load(open("results_svhn_ga/svhn_ga.json"))
ms = lambda v: f"{np.mean(v):.3f}±{np.std(v):.3f}"
for kind in ("rand", "cls"):
    print("=====", kind)
    for nm in ("original", "retrain"):
        k = [r["scen"][kind][nm] for r in R]
        print(f"{nm:10s} test {ms([a['test_acc'] for a in k])} forget {ms([a['forget_acc'] for a in k])} fcls_test {ms([a['test_forgetclass_acc'] for a in k])} other {ms([a['test_otherclass_acc'] for a in k])}" + (f" MIA {ms([a['mia_auc'] for a in k])}" if kind == "rand" else ""))
    for grp in ("stop", "stop_repair"):
        for i, lr in enumerate((1e-3, 3e-3, 1e-2)):
            k = [r["scen"][kind][grp][i] for r in R]
            print(f"{grp:12s} lr {lr:g} steps {ms([a['steps'] for a in k])} test {ms([a['test_acc'] for a in k])} forget {ms([a['forget_acc'] for a in k])} fcls_test {ms([a['test_forgetclass_acc'] for a in k])} other {ms([a['test_otherclass_acc'] for a in k])}" + (f" MIA {ms([a['mia_auc'] for a in k])}" if kind == "rand" else "") + f" cost {np.mean([a['cost_sample_passes'] for a in k])/ np.mean([r['scen'][kind]['retrain']['cost_sample_passes'] for r in R]):.4f}x retrain")
