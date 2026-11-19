"""SISA shard trainer: one FL client = one SISA shard, run on that client's own machine.

usage: python sisa_shard.py <shard_id 0|1> <out_dir> [seeds=0,1,2]

Data/partition identical to ~/FL/NonIID (CIFAR-10, Dirichlet alpha=0.3, 2 partitions,
80/20 local split, seed 42). Shard 0 = desktop, shard 1 = Mac.
Per seed it writes (in out_dir/seed<k>/):
  ckpt_<r>.pt        weights after slice r (r=1..R), cumulative slices as in SISA
  final.pt           = ckpt_R
  timing.json        seconds / sample-passes of the full schedule and of each deletion
  del_single.pt      shard retrained after deleting ONE sample located in slice R-1
  del_5pct.pt        shard retrained after deleting 5% random samples (hits slice 1 -> ~full retrain)
  deleted_idx.json   which local train indices were deleted
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

R, E, LR, BS = 5, 6, 0.01, 32  # slices, epochs per slice, lr, batch size


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5); self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120); self.fc2 = nn.Linear(120, 84); self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))); x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 400)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def to_tensors(ds):
    x = np.stack([np.asarray(i, dtype=np.float32) for i in ds["img"]]) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds["label"])


def load_shard(pid):
    part = DirichletPartitioner(num_partitions=2, partition_by="label", alpha=0.3,
                                min_partition_size=1000, self_balancing=False, seed=42)
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": part})
    s = fds.load_partition(pid).train_test_split(test_size=0.2, seed=42)
    return to_tensors(s["train"])


def train_slices(x, y, order, start_slice, w0, seed, tag, timing):
    """Train slices start_slice..R, slice r on the first r/R of `order` (cumulative), E epochs each."""
    n = len(order)
    net = Net(); net.load_state_dict(w0); outs = {}
    for r in range(start_slice, R + 1):
        idx = order[: n * r // R]
        dl = DataLoader(TensorDataset(x[idx], y[idx]), batch_size=BS, shuffle=True,
                        generator=torch.Generator().manual_seed(seed * 100 + r))
        opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        t0 = time.time(); net.train()
        for _ in range(E):
            for xb, yb in dl:
                opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
        timing[f"{tag}_slice{r}_s"] = time.time() - t0
        timing[f"{tag}_slice{r}_samples"] = len(idx) * E
        outs[r] = {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  [{tag}] slice {r}/{R} n={len(idx)} {timing[f'{tag}_slice{r}_s']:.1f}s", flush=True)
    return outs


def main():
    pid, out = int(sys.argv[1]), sys.argv[2]
    seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    torch.set_num_threads(os.cpu_count() or 4)
    x, y = load_shard(pid); n = len(y)
    print(f"shard {pid}: {n} train samples, labels {torch.bincount(y, minlength=10).tolist()}", flush=True)
    for seed in seeds:
        d = f"{out}/seed{seed}"; os.makedirs(d, exist_ok=True)
        torch.manual_seed(42 + seed); w_init = {k: v.clone() for k, v in Net().state_dict().items()}
        # slice order has its own generator seed, independent of the deletion seed
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7919)).tolist()
        pos = {i: p for p, i in enumerate(order)}
        slice_of = lambda i: pos[i] * R // n + 1   # 1-based slice that first contains sample i
        timing = {}
        ck = train_slices(x, y, order, 1, w_init, seed, "full", timing)
        for r, w in ck.items(): torch.save(w, f"{d}/ckpt_{r}.pt")
        torch.save(ck[R], f"{d}/final.pt")
        timing["full_s"] = sum(timing[f"full_slice{r}_s"] for r in range(1, R + 1))
        timing["full_samples"] = sum(timing[f"full_slice{r}_samples"] for r in range(1, R + 1))
        rng = np.random.RandomState(seed + 13)
        late = [i for i in range(n) if slice_of(i) == R - 1]
        del_single = [int(rng.choice(late))]
        del_5pct = [int(i) for i in rng.choice(n, n // 20, replace=False)]
        json.dump({"single": del_single, "5pct": del_5pct, "single_slice": slice_of(del_single[0]),
                   "5pct_min_slice": min(slice_of(i) for i in del_5pct)}, open(f"{d}/deleted_idx.json", "w"))
        for name, dele in (("single", del_single), ("5pct", del_5pct)):
            dset = set(dele); first = min(slice_of(i) for i in dele)
            new_order = [i for i in order if i not in dset]
            w0 = w_init if first == 1 else ck[first - 1]   # checkpoint just before first affected slice
            t0 = time.time(); o = train_slices(x, y, new_order, first, w0, seed, f"del_{name}", timing)
            timing[f"del_{name}_s"] = time.time() - t0
            timing[f"del_{name}_first_slice"] = first
            timing[f"del_{name}_samples"] = sum(timing[f"del_{name}_slice{r}_samples"] for r in range(first, R + 1))
            torch.save(o[R], f"{d}/del_{name}.pt")
        json.dump(timing, open(f"{d}/timing.json", "w"), indent=1)
        print(f"seed {seed} done: full {timing['full_s']:.0f}s single {timing['del_single_s']:.0f}s "
              f"5pct {timing['del_5pct_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
