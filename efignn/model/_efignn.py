import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree

from efignn.model._base import _BaseLightningModel

__all__ = ['EFIGNNConv', 'EFIGNN']


class EFIGNNConv(MessagePassing):

    def __init__(
        self,
        channels: int
    ):
        super().__init__(aggr='add')
        self._linear = nn.Linear(channels, channels, bias=False)
        self._bias = nn.Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        self._linear.reset_parameters()
        self._bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        edge_index: torch.Tensor
    ):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self._linear(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        out = out * x0
        out = out + self._bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class EFIGNN(_BaseLightningModel):

    def __init__(
        self,
        architecture: list,
        dropout: float = 0,
        batchnorm: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0,
        skip_conn: bool = False,
        apply_output_layer: bool = True
    ) -> None:
        
        super().__init__(lr, weight_decay)

        self.save_hyperparameters()

        n_layers = len(architecture) - 2

        self._layers = nn.ModuleList()

        # self._layers.append(nn.Linear(architecture[0], architecture[1]))
        self._layers.append(GCNConv(architecture[0], architecture[1]))
        if batchnorm:
            self._layers.append(nn.BatchNorm1d(architecture[1]))
        self._layers.append(nn.Dropout(p=dropout))
        for i in range(n_layers - 1):
            self._layers.append(EFIGNNConv(architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ):
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if len(xs) == 1:
                    x = layer(x, edge_index)
                else:
                    if self._skip_conn:
                        # x = layer(x, xs[0], edge_index) + sum(xs[1:])
                        x = layer(x, xs[1], edge_index) + sum(xs[1:])
                    else:
                        # x = layer(x, xs[0], edge_index)
                        x = layer(x, xs[1], edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)
        
        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat
