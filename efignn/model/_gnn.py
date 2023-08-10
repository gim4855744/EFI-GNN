import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, GCN2Conv, GATv2Conv, AntiSymmetricConv

from efignn.model._base import _BaseLightningModel

__all__ = ['GCN', 'GAT', 'GCN2', 'GAT2', 'ASDGN']


class GCN(_BaseLightningModel):

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
        for i in range(n_layers):
            self._layers.append(GCNConv(architecture[i], architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            self._layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if self._skip_conn:
                    x = layer(x, edge_index) + sum(xs[1:])
                else:
                    x = layer(x, edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)
        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat


class GAT(_BaseLightningModel):

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
        for i in range(n_layers):
            self._layers.append(GATConv(architecture[i], architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            self._layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if self._skip_conn:
                    x = layer(x, edge_index) + sum(xs[1:])
                else:
                    x = layer(x, edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)

        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat


class GCN2(_BaseLightningModel):

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
        for i in range(n_layers):
            if i == 0:
                self._layers.append(GCNConv(architecture[i], architecture[i + 1]))
            else:
                self._layers.append(GCN2Conv(architecture[i + 1], alpha=0.2, theta=0.5, layer=i + 1))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            self._layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if len(xs) == 1:  # first layer
                    x = layer(x, edge_index)
                else:
                    if self._skip_conn:
                        x = layer(x, xs[1], edge_index) + sum(xs[1:])
                    else:
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


class GAT2(_BaseLightningModel):

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
        for i in range(n_layers):
            self._layers.append(GATv2Conv(architecture[i], architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            self._layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for layer in self._layers:
            if isinstance(layer, MessagePassing):
                xs.append(x)
                if self._skip_conn:
                    x = layer(x, edge_index) + sum(xs[1:])
                else:
                    x = layer(x, edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)
        
        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat


class ASDGN(_BaseLightningModel):

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
        for i in range(n_layers):
            if i == 0:
                self._layers.append(GCNConv(architecture[i], architecture[i + 1]))
            else:
                self._layers.append(AntiSymmetricConv(architecture[i + 1]))
            if batchnorm:
                self._layers.append(nn.BatchNorm1d(architecture[i + 1]))
            self._layers.append(nn.LeakyReLU(negative_slope=0.2))
            if dropout > 0:
                self._layers.append(nn.Dropout(p=dropout))
        self._output_layer = nn.Linear(sum(architecture[1: -1]), architecture[-1])
        
        self._skip_conn = skip_conn
        self._apply_output_layer = apply_output_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for layer in self._layers:
            if layer.__class__.__name__ == 'AntiSymmetricConv' or isinstance(layer, MessagePassing):
                xs.append(x)
                if len(xs) == 1:  # first layer
                    x = layer(x, edge_index)
                else:
                    if self._skip_conn:
                        x = layer(x, edge_index) + sum(xs[1:])
                    else:
                        x = layer(x, edge_index)
            else:
                x = layer(x)
        xs.append(x)
        xs = torch.concat(xs[1:], dim=1)
        
        if self._apply_output_layer:
            y_hat = self._output_layer(xs)
        else:
            y_hat = xs

        return y_hat
