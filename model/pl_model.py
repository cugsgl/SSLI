import pytorch_lightning as pl
import torch 
import torch.nn as nn
from metric.loss import mask_loss
class LitModel(pl.LightningModule):
    def __init__(self, model, mean, std):
        super(LitModel, self).__init__()
        self.model = model
        self.loss = mask_loss(mean, std)
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x  = self(x)
        loss = self.loss(x,y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x  = self(x)
        loss = self.loss(x,y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer