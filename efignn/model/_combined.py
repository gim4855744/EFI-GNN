import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from efignn.model._base import _BaseLightningModel
from efignn.model._gnn import GAT, GAT2, ASDGN
from efignn.model._efignn import EFIGNN

__all__ = ['GATnEFIGNN', 'GAT2nEFIGNN', 'ASDGNnEFIGNN']

class GATnEFIGNN(_BaseLightningModel):

    def __init__(
        self,
        gnn_hparam: dict,
        efignn_hparam: dict
    ) -> None:
        
        super().__init__(gnn_hparam['lr'], gnn_hparam['weight_decay'])

        self.save_hyperparameters()

        self._gnn = GAT(**gnn_hparam, apply_output_layer=False)
        self._efignn = EFIGNN(**efignn_hparam, apply_output_layer=False)

        # self._output_layer = GCNConv(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]),
        #                              gnn_hparam['architecture'][-1])
        self._output_layer = Linear(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]), gnn_hparam['architecture'][-1])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        gnn_y_hat = self._gnn(x, edge_index)
        efignn_y_hat = self._efignn(x, edge_index)
        y_hat = torch.concat([gnn_y_hat, efignn_y_hat], dim=1)
        # y_hat = self._output_layer(y_hat, edge_index)
        y_hat = self._output_layer(y_hat)
        return y_hat


class GAT2nEFIGNN(_BaseLightningModel):

    def __init__(
        self,
        gnn_hparam: dict,
        efignn_hparam: dict
    ) -> None:
        
        super().__init__(gnn_hparam['lr'], gnn_hparam['weight_decay'])

        self.save_hyperparameters()

        self._gnn = GAT2(**gnn_hparam, apply_output_layer=False)
        self._efignn = EFIGNN(**efignn_hparam, apply_output_layer=False)

        # self._output_layer = GCNConv(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]),
        #                              gnn_hparam['architecture'][-1])
        self._output_layer = Linear(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]), gnn_hparam['architecture'][-1])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        gnn_y_hat = self._gnn(x, edge_index)
        efignn_y_hat = self._efignn(x, edge_index)
        y_hat = torch.concat([gnn_y_hat, efignn_y_hat], dim=1)
        # y_hat = self._output_layer(y_hat, edge_index)
        y_hat = self._output_layer(y_hat)
        return y_hat


class ASDGNnEFIGNN(_BaseLightningModel):

    def __init__(
        self,
        gnn_hparam: dict,
        efignn_hparam: dict
    ) -> None:
        
        super().__init__(gnn_hparam['lr'], gnn_hparam['weight_decay'])

        self.save_hyperparameters()

        self._gnn = ASDGN(**gnn_hparam, apply_output_layer=False)
        self._efignn = EFIGNN(**efignn_hparam, apply_output_layer=False)

        # self._output_layer = GCNConv(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]),
        #                              gnn_hparam['architecture'][-1])
        self._output_layer = Linear(sum(gnn_hparam['architecture'][1: -1]) + sum(efignn_hparam['architecture'][1: -1]), gnn_hparam['architecture'][-1])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        gnn_y_hat = self._gnn(x, edge_index)
        efignn_y_hat = self._efignn(x, edge_index)
        y_hat = torch.concat([gnn_y_hat, efignn_y_hat], dim=1)
        # y_hat = self._output_layer(y_hat, edge_index)
        y_hat = self._output_layer(y_hat)
        return y_hat
