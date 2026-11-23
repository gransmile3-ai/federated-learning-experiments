# Unlearning across datasets and models (Mac, MPS)

4 datasets (MNIST, SVHN, CIFAR-10, CIFAR-100) x 4 models (small CNN, MLP, ResNet-18 with CIFAR stem, tiny ViT), one identical recipe: 6 epochs, no augmentation. Seeds: CNN/MLP/ViT 3, ResNet-18 1 (about 1 h per combo).
Forget targets: all train samples of class 3, and 1000 random train samples. Methods: retrain (gold standard), GA (plain SGD ascent in eval mode, stop at forget loss >= ln(#classes), lr 1e-3 / 3e-3; tables use 1e-3), GA + repair (1 epoch on 5000 retain samples), SISA (S=5 shards, R=5 slices, 2 epochs per cumulative slice, retrain only affected shards). Cost = sample passes relative to the retrain.

Files: `mu_common.py` (data/models), `multi_run.py` (one combo), `run_all.py` (driver), `aggregate.py` -> `summary.md`, `multi_cls.png`, `multi_rand.png`; raw `results/*.json`.
Run: `python run_all.py` (skips finished combos), then `python aggregate.py`.

Findings (full tables in `summary.md`):
- SISA forgets exactly in every combo (class acc 0.000), but with this budget the sharded ensemble is less accurate (CIFAR-100 CNN test 0.044 vs 0.222 retrain) and costs 100% of a retrain because IID shards all contain the forget data.
- GA alone: cost 1-4% of retrain, forgets the class, but test acc drops; ResNet-18 collapses on SVHN class-3 (test 0.062) and on random-1000 (mean test 0.14).
- GA + repair: 2-14% cost, test acc recovered, but class accuracy returns to 0.3-0.9 on MNIST/SVHN (ResNet-18 SVHN 0.873).
- On CIFAR-10/100 the class-3 "forget acc 0.000" is weak evidence: the original model barely recognises that class (CIFAR-100 CNN 0.058). Only ResNet-18 CIFAR-10 (original 0.715) is informative.
Caveats: single recipe per model, no per-model tuning of GA lr/stop rule, ResNet-18 has one seed, SISA undertrained per shard, MIA only logged for random targets.

## EECC evaluation (Mercuri et al. 2022, Sec. 3)
`eecc.py <results_dir> <out_prefix>` computes the four properties against the naive retrain: Efficiency (retrain time / unlearn time, also sample passes), Effectiveness (|test acc diff|), Certifiability (SAPE of accuracy on the forgotten data, Eq. 3.6), Consistency (test-prediction agreement, KL(retrain || unlearned), parameter L2).
- `eecc_noconsistency.md`: all 16 combos, from the original sweep (`results/`); consistency columns empty.
- `eecc_cnn_mlp.md`: CNN and MLP, 3 seeds, rerun with consistency (`multi_run_c.py`, `run_all_c.py`, `results_c/`). Accuracies reproduce the original sweep.
- Not measured: Consistency for ViT and ResNet-18 (weights were not saved in the first sweep).
Reading notes: (1) the original model itself agrees with the retrain on only 0.69-0.77 of test predictions (independently trained networks differ), so agreement must be compared against that floor; (2) GA changes parameters little (L2 ~ same as the original) but its outputs differ a lot from the retrain (KL ~3 on random targets); (3) SISA's consistency is measured against a single-model retrain, while its exactness is relative to a retrained shard ensemble, so those numbers are not a test of SISA's exactness; (4) Certifiability for random targets is weak evidence because original and retrain nearly coincide there.
