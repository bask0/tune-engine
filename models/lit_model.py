import torch
from torch import nn
import lightning.pytorch as pl


class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, hidden_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.layer_1 = nn.Linear(28 * 28, self.hparams.hidden_size)
        self.layer_2 = nn.Linear(self.hparams.hidden_size, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
