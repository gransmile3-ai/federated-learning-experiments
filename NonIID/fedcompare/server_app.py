"""fedcompare: A Flower / PyTorch FedProx app."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fedcompare.task import Net, load_centralized_dataset, test

# 결과 저장 위치 (SuperLink/ServerApp이 도는 AWS 서버 기준: ~/FedProx)
OUT_DIR = Path.home() / "NonIID"

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    mu: float = context.run_config["proximal-mu"]

    name: str = context.run_config["strategy"]
    tag: str = context.run_config["run-tag"]
    torch.manual_seed(42)  # 모든 실험이 같은 초기 가중치에서 시작
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # FedProx: 서버 집계는 FedAvg와 같고, train config에 "proximal-mu"를 실어 보냄
    if name == "fedprox":
        strategy = FedProx(fraction_evaluate=fraction_evaluate, proximal_mu=mu)
    else:
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    OUT_DIR.mkdir(exist_ok=True)
    if context.run_config["save-model"]:
        print("\nSaving final model to disk...")
        torch.save(result.arrays.to_torch_state_dict(), OUT_DIR / f"{tag}_model.pt")

    def plain(d):
        return {int(k): {kk: float(vv) for kk, vv in dict(v).items()} for k, v in d.items()}

    summary = {
        "strategy": name,
        "tag": tag,
        "partition": context.run_config["partition"],
        "dirichlet_alpha": float(context.run_config["dirichlet-alpha"]),
        "proximal_mu": mu if name == "fedprox" else 0.0,
        "lr": lr,
        "local_epochs": context.run_config["local-epochs"],
        "rounds": num_rounds,
        "server_eval": plain(result.evaluate_metrics_serverapp),
        "client_eval": plain(result.evaluate_metrics_clientapp),
        "client_train": plain(result.train_metrics_clientapp),
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_DIR}")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
