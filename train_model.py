import os 
from optparse import OptionParser

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.nn.functional import one_hot
from torchmetrics.classification import BinaryAUROC

from torcheeg import transforms
from torcheeg.datasets import NumpyDataset
from torcheeg.models import EEGNet

from torchinfo import summary

import config
from trainer import train, validate
from models import *

# load cached datasets
import munging
import numpy as np
from scipy.stats import zscore

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, sz_thresh):
        self.X = zscore(X, axis=1).astype(np.float32)
        self.y = np.array(y >= sz_thresh).astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main(size, sz_thresh = 2.0):
    save_path_X_trn = "./cache/X_trn%s.npy" % size
    save_path_y_trn = "./cache/y_trn%s.npy" % size
    save_path_X_dev = "./cache/X_dev%s.npy" % size
    save_path_y_dev = "./cache/y_dev%s.npy" % size

    # load train set
    with open(save_path_X_trn, "rb") as f:
        X_trn = np.load(f)
    with open(save_path_y_trn, "rb") as f:
        y_raw_trn = np.load(f)

    # y_trn = munging.format_y_for_torcheeg(y_raw_trn)

    y_bin = (y_raw_trn > sz_thresh).astype(int)
    num_pos_samples = np.sum(y_bin)
    num_neg_samples = len(y_bin) - num_pos_samples
    class_weights = [1 / num_neg_samples, 1 / num_pos_samples]
    sample_weights = np.array([class_weights[t] for t in y_bin])

    # train_set = NumpyDataset(X = X_trn,
    #                          y = y_trn,
    #                          io_path="./io/tuh_trn%s" % size,
    #                          online_transform=transforms.Compose([
    #                             transforms.MeanStdNormalize(axis=1),
    #                             transforms.ToTensor()]),
    #                          label_transform=transforms.Compose([
    #                              transforms.Select('sz_present'),
    #                              transforms.Binary(sz_thresh)]),
    #                          num_worker=1)
    # del X_trn
    # del y_raw_trn
    # del y_trn

    train_set = CustomDataset(X_trn, y_raw_trn, sz_thresh)

    # load val set
    with open(save_path_X_dev, "rb") as f:
        X_dev = np.load(f)
    with open(save_path_y_dev, "rb") as f:
        y_raw_dev = np.load(f)
        
    # y_dev = munging.format_y_for_torcheeg(y_raw_dev)
    # val_set = NumpyDataset(X = X_dev,
    #                        y = y_dev,
    #                        io_path="./io/tuh_dev%s" % size,
    #                        online_transform=transforms.Compose([
    #                             transforms.MeanStdNormalize(axis=1),
    #                             transforms.ToTensor()]),
    #                        label_transform=transforms.Compose([
    #                            transforms.Select('sz_present'),
    #                            transforms.Binary(sz_thresh)]),
    #                        num_worker=64)
    # del X_dev
    # del y_raw_dev
    # del y_dev

    val_set = CustomDataset(X_dev, y_raw_dev, sz_thresh)

    torch_auroc = BinaryAUROC(thresholds=None)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            X = batch[0].to(device)
            y = batch[1].to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                auroc = torch_auroc(pred, y).item()
                print(f"loss: {loss:>5f}, auroc: {auroc:>.4f} [{current:>5d}/{size:>5d}]")


    def validate(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        val_loss, correct = 0, 0
        all_pred, all_y = [], []
        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(device)
                y = batch[1].to(device)

                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
                all_pred.append(pred.cpu().numpy())
                all_y.append(batch[1])
        val_loss /= num_batches
        correct /= size

        auroc = roc_auc_score(np.concatenate(all_y), np.concatenate(all_pred)[:,1])
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f}, \
                AUROC: {(auroc):>0.3f}\n")


    device = "mps"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    model = CNN(in_channels = 20, output_size=1)
    print(summary(model, input_size=(64, 20, 640)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    class_weights = []
    sampler = WeightedRandomSampler(weights = sample_weights,
                                    num_samples=int(num_pos_samples + num_neg_samples),
                                    replacement=True)
    train_loader = DataLoader(train_set, 
                              sampler = sampler,
                              batch_size = batch_size)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        validate(val_loader, model, loss_fn)
        print("Done!")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--size", dest="size")

    options, _ = parser.parse_args()

    size = "_" + options.size

    main(size)



