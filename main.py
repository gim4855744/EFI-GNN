import warnings
import logging
import argparse
import os
from dataclasses import asdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader

from efignn.dataset import *
from efignn.hparam import *
from efignn.model import *

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
parser.add_argument('--dataset', type=str, choices=DATASET_MAP.keys(), required=True)
parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), required=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--silent', action='store_false')
args = parser.parse_args()


def main():

    pl.seed_everything(args.seed)
    
    dataset = DATASET_MAP[args.dataset]()
    loader = DataLoader(dataset)
    
    ckptdir = './checkpoints/'
    ckptname = f'{args.dataset}_{args.model}_{args.seed}'
    ckptpath = os.path.join(ckptdir, f'{ckptname}.ckpt')

    checkpoint_callback = ModelCheckpoint(ckptdir, ckptname, monitor='val_loss')
    earlystop_callback = EarlyStopping('val_loss', patience=20)
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='cuda',
        devices=[2],
        max_epochs=1000,
        logger=False,
        enable_model_summary=False,
        callbacks=callbacks,
        enable_progress_bar=args.silent
    )

    if args.mode == 'train':

        if os.path.exists(ckptpath):
            os.remove(ckptpath)

        gnn_hparam = asdict(HPARAM_MAP[f'{args.dataset}_gnn'])
        efignn_hparam = asdict(HPARAM_MAP[f'{args.dataset}_efignn'])
        if args.model in ['gcn', 'gat', 'gcn2', 'gat2', 'asdgn']:
            model = MODEL_MAP[args.model](**gnn_hparam)
        elif args.model == 'efignn':
            model = MODEL_MAP[args.model](**efignn_hparam)
        else:  # combined model
            model = MODEL_MAP[args.model](gnn_hparam, efignn_hparam)
            
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    else:
        model = MODEL_MAP[args.model].load_from_checkpoint(ckptpath)
        trainer.test(model, dataloaders=loader)


if __name__ == '__main__':
    main()
