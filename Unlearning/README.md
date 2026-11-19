# FL client unlearning (forget the desktop client)
Setup: CIFAR-10, 2 clients (desktop=partition 0, Mac=partition 1), Dirichlet α=0.3, FedAvg 20 rounds × 3 local epochs, lr 0.01. Plain-PyTorch FedAvg loop on the Mac (same data/partitions/model/seeds as ~/FL/NonIID). Gold standard = retrain from the same init with the Mac client only.
Files: unlearn.py (stage 1: original, retrain, subtract, GA sweep, vanilla FedEraser), stage2.py (stopped GA + repair, FedEraser-mix), plot.py, results/results.json, compare.png, per_class.png.
Run: `cd ~/FL/NonIID && PYTHONPATH=.:$HOME/FL/Unlearning ~/FL/.venv-1/bin/python ~/FL/Unlearning/unlearn.py ~/FL/Unlearning/results` then stage2.py, plot.py.
Findings: single runs, one seed. Vanilla GA collapses the model (acc 10%, NaN loss) unless stopped early; vanilla FedEraser diverges under this label skew (NaN); blending the stored direction with a fresh one (α=0.25) works. GA+1 repair round matches retrain on forget acc/MIA at ~3 epochs vs 60.

## SISA (shard = FL client), 3 seeds
Desktop trains shard 0 on its own machine, Mac trains shard 1 (`sisa_shard.py`, R=5 slices x 6 epochs, cumulative slices, checkpoint per slice); ensemble = mean softmax, evaluated on the Mac (`sisa_eval.py`). Results: `results_sisa/sisa_results.json`; desktop copy in `Desktop\FL\Unlearning\`.
- Full ensemble: test 0.532±0.010, desktop-train acc 0.766, client MIA 0.620.
- Forget the desktop CLIENT = drop shard 0: test 0.403±0.010, desktop-train acc 0.153, MIA 0.500, retrain cost 0 (identical to the Mac-only model by construction).
- Delete 1 sample (in slice 4/5): retrain cost 0.60 of full (1.7x faster). Delete 5% random: first affected slice is 1, cost 0.95 (1.05x, no gain). 5% deleted-sample MIA AUC 0.629 -> 0.518.
Caveats: different training budget from the FedAvg run (18 epoch-equivalents per shard vs 20 rounds x 3 epochs); ensemble beats FedAvg here partly because label-skewed shards complement each other; 2 shards only.
