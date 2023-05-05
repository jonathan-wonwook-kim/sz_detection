import os 
from optparse import OptionParser

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
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

from mmapdataset import MMAPDataset, Xy_iter, get_dataset_size

def main(size, sz_thresh = 2.0):
    # parameters that might change
    device = "mps"
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, 19]).to(device))
    batch_size = 64 
    resample_training_data = False
    epochs = 20

    sig = nn.Sigmoid()

    #tensorboard stuff
    writer = SummaryWriter()
    
    print("loading train set...")
    train_set = MMAPDataset(Xy_iter("X", "trn", "first_half"),
                            Xy_iter("y", "trn", "first_half", sz_thresh),
                            size=get_dataset_size(config.WINS_PER_CACHE_FILE, 
                                "trn", "first_half"),
                            transform_fn=transforms.Compose([
                                transforms.MeanStdNormalize(axis=1),
                                transforms.ToTensor()]),
                            num_windows_per_file=config.WINS_PER_CACHE_FILE)
    print("...loaded train set.")

    print("loading val set...")
    val_set = MMAPDataset(Xy_iter("X", "dev", "first_half"),
                          Xy_iter("y", "dev", "first_half", sz_thresh),
                            size=get_dataset_size(config.WINS_PER_CACHE_FILE,
                                "dev", "first_half"),
                            transform_fn=transforms.Compose([
                                transforms.MeanStdNormalize(axis=1),
                                transforms.ToTensor()]),
                            num_windows_per_file=config.WINS_PER_CACHE_FILE)
    print("...loaded val set.")

    def train(dataloader, model, loss_fn, optimizer, epoch):
        size = len(dataloader.dataset)
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            X = batch[0].to(device)
            y = one_hot(batch[1].to(device), num_classes=2)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                pred_cpu = sig(pred)[:,1].cpu().detach().numpy()
                y_true = y[:,1].cpu().numpy()
                if len(np.unique(y_true)) ==  2:
                    auroc = roc_auc_score(y_true = y_true, 
                                         y_score = pred_cpu)
                else:
                    auroc = np.nan
                print(f"loss: {loss:>5f}, AUROC: {auroc:>.4f} [{current:>5d}/{size:>5d}]")

                n_iter = epoch * size / 6400 + batch_idx / 100
                writer.add_scalar('Loss/train', loss, n_iter)
                writer.add_scalar('AUROC/train', auroc, n_iter)

    def validate(dataloader, model, loss_fn, epoch):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        val_loss, correct = 0, 0
        # all_pred, all_y = [], []
        all_pred = torch.Tensor([]).to(device)
        all_y = torch.Tensor([]).to(device)
        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(device)
                y = one_hot(batch[1].to(device), num_classes=2)

                pred = model(X)
                val_loss += loss_fn(pred, y.float()).item()

                correct += (sig(pred).argmax(1) == y[:,1]).type(torch.float).sum().item()
                
                # all_pred.append(pred.cpu().numpy())
                # all_y.append(batch[1])
                all_pred = torch.cat([all_pred, pred], 0)
                all_y = torch.cat([all_y, y], 0)
        val_loss /= num_batches
        correct /= size

        pred_cpu = sig(all_pred)[:,1].cpu().detach().numpy()
        auroc = roc_auc_score(y_true = all_y[:,1].cpu().numpy(),
                              y_score = pred_cpu)
        print(f"Test Error:\nAvg loss: {val_loss:>8f}, AUROC: {(auroc):>0.3f}\n")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUROC/val', auroc, epoch)
        # import IPython; IPython.embed()

    model = CNN(in_channels = 20)
    # print(summary(model, input_size=(64, 20, 640)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    class_weights = []
    if resample_training_data:
        sampler = WeightedRandomSampler(weights = sample_weights,
                            num_samples=int(num_pos_samples + num_neg_samples),
                            replacement=True)
        train_loader = DataLoader(train_set, 
                            sampler = sampler,
                            batch_size = batch_size)
    else:
        train_loader = DataLoader(train_set, 
                            shuffle=True, 
                            batch_size=batch_size)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, epoch=t)
        validate(val_loader, model, loss_fn, epoch=t)
        print("Done!")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--size", dest="size", default="")

    options, _ = parser.parse_args()

    main(options.size)



