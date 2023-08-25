# Explicit Feature Interaction-award Graph Neural Network (EFI-GNN)

EFI-GNN is an graph neural network (GNN) that explicitly learn arbitrary-order feature interactions.

## Requirements
- pytorch
- lightning
- torch-geometric
- ogb

## Quick Start

To train the EFI-GNN:
```shell
python main.py --mode=train --dataset=<dataset name> --model=efignn
```

To test the EFI-GNN:
```shell
python main.py --mode=test --dataset=<dataset name> --model=efignn
```
