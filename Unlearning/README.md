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

## SVHN SISA (dataset from Bourtoule et al. 2021), Mac, 3 seeds
`svhn_sisa.py` -> `results_svhn/svhn_sisa.json`, plot `plot_svhn.py` -> `svhn_sisa.png`. SVHN train 73,257 / test 26,032, same small CNN, IID shards, R=5 cumulative slices x 3 epochs, S in {5,10,20}; baseline = one model, 9 epochs on all data (equal sample-pass budget). Deletions: 1 / 10 / 100 random samples; cost = sample passes retrained (affected shards only, from the checkpoint before their first affected slice).
- Test acc: single 0.888, S=5 0.868, S=10 0.830, S=20 0.348±0.106 (undertrained: 3.6k samples/shard, 3 epochs).
- Speed-up vs naive retrain: delete 1 -> 6.1x / 12.2x / 24.5x (S=5/10/20); delete 10 -> 1.2x / 1.6x / 2.7x; delete 100 -> ~1.0x (all shards hit).
- Check, seed 0 only (`SISA_E=10 SISA_S=20`, `results_svhn/svhn_sisa_S20_E10.json`): S=20 with 10 epochs per slice reaches 0.857, so the S=20 collapse is a training-budget artifact. (Its speed-ups are relative to a 9-epoch baseline, so not comparable to the table.)
Caveats: small CNN, uniform IID shards, cost counted in sample passes not wall time.
