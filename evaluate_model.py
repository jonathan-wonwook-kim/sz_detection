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

# from mmapdataset import MMAPDataset, Xy_iter, get_dataset_size

import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
import pandas as pd 

def main(subset, sz_thresh = 2.0):
    # parameters that might change
    device = "mps"
    model_path = "./" + config.MODEL_NAME

    # train data is about 5% positive examples, hence weight is 1:19
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, 19]).to(device))
    batch_size = 64 
    resample_training_data = False
    epochs = 50

    sig = nn.Sigmoid()

    #tensorboard stuff
    writer = SummaryWriter()
    
    print("loading eval set...")
    which_set = "eval"
    save_path_X_eval = "./cache/X_eval_%s.npy" % subset
    save_path_y_eval = "./cache/y_eval_%s.npy" % subset
    with open(save_path_X_eval, "rb") as f:
        X_eval = np.load(f)
    with open(save_path_y_eval, "rb") as f:
        y_raw_eval = np.load(f)

    # val_set = CustomDataset(X_dev, y_raw_dev, sz_thresh)
    y_eval = munging.format_y_for_torcheeg(y_raw_eval)
    eval_set = NumpyDataset(X = X_eval,
                            y = y_eval,
                            io_mode="pickle",
                            io_path="./io/tuh_eval_%s" % subset,
                            online_transform=transforms.Compose([
                                 transforms.MeanStdNormalize(axis=1),
                                 transforms.ToTensor()]),
                            label_transform=transforms.Compose([
                                transforms.Select('sz_present'),
                                transforms.Binary(sz_thresh)]),
                            num_worker=64)
    print("...loaded eval set.")

    def evaluate(dataloader, model, loss_fn):
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
                
                all_pred = torch.cat([all_pred, pred], 0)
                all_y = torch.cat([all_y, y], 0)
        val_loss /= num_batches
        correct /= size

        pred_cpu = sig(all_pred)[:,1].cpu().detach().numpy()
        auroc = roc_auc_score(y_true = all_y[:,1].cpu().numpy(),
                              y_score = pred_cpu)
        print(f"Eval Error:\nAvg loss: {val_loss:>8f}, AUROC: {(auroc):>0.3f}\n")

        y = all_y[:,1].cpu().numpy()
        scores = pred_cpu
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)

        
        df_roc = pd.DataFrame(data={"fpr": fpr, "tpr": tpr})
        import IPython; IPython.embed()

    model = CNN(in_channels = 20)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # print(summary(model, input_size=(64, 20, 640)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    evaluate(eval_loader, model, loss_fn)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--subset", dest="subset", default="")

    options, _ = parser.parse_args()

    main(options.subset)



