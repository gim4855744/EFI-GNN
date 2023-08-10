from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

__all__ = ['DATASET_MAP']


def fetch_cora(root='./data/'):
    dataset = Planetoid(root, 'cora', split='full')
    return dataset


def fetch_citeseer(root='./data/'):
    dataset = Planetoid(root, 'citeseer', split='full')
    return dataset


def fetch_pubmed(root='./data/'):
    dataset = Planetoid(root, 'pubmed', split='full')
    return dataset


def fetch_ogbn_arxiv(root='./data/'):
    dataset = PygNodePropPredDataset('ogbn-arxiv', root)
    split_idx = dataset.get_idx_split()
    dataset._data['y'] = dataset._data['y'].reshape(-1)
    dataset._data['train_mask'] = split_idx['train']
    dataset._data['val_mask'] = split_idx['valid']
    dataset._data['test_mask'] = split_idx['test']
    return dataset


def fetch_ogbn_mag(root='./data/'):
    dataset = PygNodePropPredDataset('ogbn-mag', root)
    split_idx = dataset.get_idx_split()
    dataset._data['edge_index'] = dataset._data['edge_index_dict'][('paper', 'cites', 'paper')]
    dataset._data['x'] = dataset._data['x_dict']['paper']
    dataset._data['y'] = dataset._data['y_dict']['paper'].reshape(-1)
    dataset._data['train_mask'] = split_idx['train']['paper']
    dataset._data['val_mask'] = split_idx['valid']['paper']
    dataset._data['test_mask'] = split_idx['test']['paper']
    return dataset


DATASET_MAP = {
      'cora': fetch_cora,
      'citeseer': fetch_citeseer,
      'pubmed': fetch_pubmed,
      'ogbn-arxiv': fetch_ogbn_arxiv,
      'ogbn-mag': fetch_ogbn_mag
}
