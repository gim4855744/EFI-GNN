from ._gnn import *
from ._efignn import *
from ._combined import *

MODEL_MAP = {
    'gcn': GCN,
    'gat': GAT,
    'gcn2': GCN2,
    'gat2': GAT2,
    'asdgn': ASDGN,
    'efignn': EFIGNN,
    'gatnefignn': GATnEFIGNN,
    'gat2nefignn': GAT2nEFIGNN,
    'asdgnnefignn': ASDGNnEFIGNN
}
