from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning.pytorch as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, fold_idx: int = 0, num_folds: int = 6):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        MNIST(root=".", train=True, download=True)

    def setup(self, stage=None):
        mnist_full = MNIST(root=".", train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
