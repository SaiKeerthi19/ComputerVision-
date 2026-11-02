# Homework 4 – Design notes

## Data pipeline
* `torchvision.datasets.CIFAR10(download=True)` downloads once into `./cifar10` (git-ignored).
* Transform: `ToTensor()` (0–1) then `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))` (−1–1).
* `DataLoader(batch_size=4, shuffle=True)`; `next(iter(loader))` gives one random mini-batch
  of shape (4, 3, 32, 32).

## Network (PyTorch CIFAR-10 tutorial)

| Layer | Output shape |
|-------|--------------|
| input | 3 × 32 × 32 |
| conv1 (3→6, 5×5) + ReLU | 6 × 28 × 28 |
| max-pool 2×2 | 6 × 14 × 14 |
| conv2 (6→16, 5×5) + ReLU | 16 × 10 × 10 |
| max-pool 2×2 | 16 × 5 × 5 |
| flatten | 400 |
| fc1 + ReLU | 120 |
| fc2 + ReLU | 84 |
| fc3 | 10 (one score per class) |

About 62k parameters (`cifar_net_2epoch.pth` ≈ 250 KB).

## Training
* Loss: `CrossEntropyLoss`; optimizer: `SGD(lr=0.001, momentum=0.9)`; 2 epochs = 25,000 mini-batches.
* Weights saved with `torch.save(net.state_dict(), ...)`, loaded with
  `torch.load(path, map_location="cpu")` so they work on any machine.

## Weights (Q3)
`net.conv1.weight` (6, 3, 5, 5) and `net.conv2.weight` (16, 6, 5, 5), `.detach().clone()`, bias excluded.

## Hyperparameter sweep (Q4)
Same training for lr ∈ {0.01, 0.001, 0.0001} with `torch.manual_seed(0)` so runs start equally.
Every 2,000 iterations: average loss, and error on 1,000 random train / test images
(`torch.randperm` + `Subset`) for speed.
