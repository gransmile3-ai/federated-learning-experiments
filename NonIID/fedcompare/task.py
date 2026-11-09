"""fedcompare: FedAvg vs FedProx, IID / non-IID (Dirichlet) CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int,
              partition: str = "dirichlet", alpha: float = 0.3):
    """Load partition CIFAR10 data (IID or Dirichlet label-skew; seeded so all nodes agree)."""
    global fds
    if fds is None:
        if partition == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions, partition_by="label",
                alpha=alpha, min_partition_size=1000, self_balancing=False, seed=42,
            )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    part = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    split = part.train_test_split(test_size=0.2, seed=42)
    split = split.with_transform(apply_transforms)
    trainloader = DataLoader(split["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(split["test"], batch_size=batch_size)
    return trainloader, testloader


def label_hist(partition_id: int, num_partitions: int, partition: str, alpha: float):
    """Label counts of this node's raw partition (for reporting)."""
    load_data(partition_id, num_partitions, 32, partition, alpha)
    labels = fds.load_partition(partition_id)["label"]
    return [labels.count(c) for c in range(10)]


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device, proximal_mu=0.0):
    """Train the model on the training set (FedProx: + mu/2 * ||w - w_global||^2)."""
    net.to(device)  # move model to GPU if available
    # 학습 시작 전 전역 가중치 사본 (근접항 기준점, 기울기 계산 제외)
    global_params = [p.detach().clone() for p in net.parameters()]
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            if proximal_mu > 0:
                proximal_term = sum(
                    (w - w_g).pow(2).sum() for w, w_g in zip(net.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
