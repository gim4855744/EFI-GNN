import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F


class _BaseLightningModel(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        weight_decay: float
    ) -> None:
        super().__init__()
        self._lr = lr
        self._weight_decay = weight_decay
        self._y_hat = []
        self._y_true = []

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_index) -> torch.Tensor:
        self.train()
        y_hat = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_index) -> torch.Tensor:
        self.eval()
        y_hat = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(y_hat[batch.val_mask], batch.y[batch.val_mask])
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_index) -> None:
        self.eval()
        y_hat = self(batch.x, batch.edge_index)[batch.test_mask].argmax(dim=1)
        y_true = batch.y[batch.test_mask]
        self._y_hat.append(y_hat)
        self._y_true.append(y_true)
    
    def on_test_epoch_end(self) -> None:
        y_hat = torch.concat(self._y_hat, dim=0)
        y_true = torch.concat(self._y_true, dim=0)
        acc = (y_hat == y_true).to(torch.float32).mean()
        print(acc.item())
        self._y_hat.clear()
        self._y_true.clear()
