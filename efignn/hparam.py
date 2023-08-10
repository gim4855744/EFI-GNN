from dataclasses import dataclass

__all__ = ['HPARAM_MAP']


@dataclass(frozen=True)
class _HyperParameters:
    architecture: list
    dropout: float
    batchnorm: bool
    lr: float
    weight_decay: float
    skip_conn: bool


_CORA_GNN_HPARAM = _HyperParameters(
    architecture=[1433, 128, 128, 128, 128, 7],
    dropout=0.9,
    batchnorm=False,
    lr=1e-3,
    weight_decay=1e-2,
    skip_conn=False
)
_CORA_EFIGNN_HPARAM = _HyperParameters(
    architecture=[1433, 128, 128, 128, 128, 128, 7],
    dropout=0.9,
    batchnorm=False,
    lr=1e-3,
    weight_decay=1e-2,
    skip_conn=False
)
# _CORA_EFIGNN_HPARAM = _HyperParameters(
#     architecture=[1433, 128, 128, 128, 7],
#     dropout=0.9,
#     batchnorm=False,
#     lr=1e-3,
#     weight_decay=1e-2,
#     skip_conn=False
# )
_CITESEER_GNN_HPARAM = _HyperParameters(
    architecture=[3703, 128, 128, 128, 128, 6],
    dropout=0.9,
    batchnorm=False,
    lr=1e-3,
    weight_decay=1e-2,
    skip_conn=False
)
_CITESEER_EFIGNN_HPARAM = _HyperParameters(
    architecture=[3703, 128, 128, 128, 128, 128, 6],
    dropout=0.9,
    batchnorm=False,
    lr=1e-3,
    weight_decay=1e-2,
    skip_conn=False
)
# _CITESEER_EFIGNN_HPARAM = _HyperParameters(
#     architecture=[3703, 128, 128, 128, 6],
#     dropout=0.9,
#     batchnorm=False,
#     lr=1e-3,
#     weight_decay=1e-2,
#     skip_conn=False
# )
_PUBMED_GNN_HPARAM = _HyperParameters(
    architecture=[500, 1024, 1024, 1024, 1024, 3],
    dropout=0.85,
    batchnorm=True,
    lr=1e-3,
    weight_decay=1e-3,
    skip_conn=True
)
# _PUBMED_EFIGNN_HPARAM = _HyperParameters(
#     architecture=[500, 1024, 1024, 1024, 3],
#     dropout=0.85,
#     batchnorm=True,
#     lr=1e-3,
#     weight_decay=1e-3,
#     skip_conn=True
# )
_PUBMED_EFIGNN_HPARAM = _HyperParameters(
    architecture=[500, 128, 128, 128, 128, 128, 3],
    dropout=0.85,
    batchnorm=False,
    lr=1e-3,
    weight_decay=1e-3,
    skip_conn=False
)
_OGBNARXIV_GNN_HPARAM = _HyperParameters(
    architecture=[128, 128, 128, 40],
    dropout=0.3,
    batchnorm=True,
    lr=1e-2,
    weight_decay=0,
    skip_conn=False
)
_OGBNARXIV_EFIGNN_HPARAM = _HyperParameters(
    architecture=[128, 128, 128, 40],
    dropout=0.3,
    batchnorm=True,
    lr=1e-2,
    weight_decay=0,
    skip_conn=False
)
_OGBNMAG_GNN_HPARAM = _HyperParameters(
    architecture=[128, 128, 128, 349],
    dropout=0.3,
    batchnorm=True,
    lr=1e-2,
    weight_decay=0,
    skip_conn=False
)
_OGBNMAG_EFIGNN_HPARAM = _HyperParameters(
    architecture=[128, 128, 128, 349],
    dropout=0.3,
    batchnorm=True,
    lr=1e-2,
    weight_decay=0,
    skip_conn=False
)

HPARAM_MAP = {
    'cora_gnn': _CORA_GNN_HPARAM,
    'cora_efignn': _CORA_EFIGNN_HPARAM,
    'citeseer_gnn': _CITESEER_GNN_HPARAM,
    'citeseer_efignn': _CITESEER_EFIGNN_HPARAM,
    'pubmed_gnn': _PUBMED_GNN_HPARAM,
    'pubmed_efignn': _PUBMED_EFIGNN_HPARAM,
    'ogbn-arxiv_gnn': _OGBNARXIV_GNN_HPARAM,
    'ogbn-arxiv_efignn': _OGBNARXIV_EFIGNN_HPARAM,
    'ogbn-mag_gnn': _OGBNMAG_GNN_HPARAM,
    'ogbn-mag_efignn': _OGBNMAG_EFIGNN_HPARAM
}
