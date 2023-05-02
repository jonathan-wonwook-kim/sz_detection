import os
import numpy as np
import mne
import pandas as pd
import tqdm
import glob
import gc

import torcheeg
from torcheeg.datasets import NumpyDataset
from torcheeg import transforms

import environ
import munging

if os.path.isfile("X_dev.npy") and os.path.isfile("y_dev.npy"):
    with open("X_dev.npy", "rb") as f:
        X_dev = np.load(f)
    with open("y_dev.npy", "rb") as f:
        y_raw_dev = np.load(f)
else:
    eegpaths_dev = glob.glob(os.path.join(environ.TUH_DEV_DATA_DIR, "*/*/*/*.edf"))
    annpaths_dev = glob.glob(os.path.join(environ.TUH_DEV_DATA_DIR, "*/*/*/*.csv"))
    binannpaths_dev = glob.glob(os.path.join(environ.TUH_DEV_DATA_DIR, "*/*/*/*.csv_bi"))

    subset_eegpaths_dev = np.sort(eegpaths_dev)[:64]
    subset_annpaths_dev = []
    subset_binannpaths_dev = []
    for f in subset_eegpaths_dev:
        subset_annpaths_dev.append(f.split(".edf")[0] + ".csv")
        subset_binannpaths_dev.append(f.split(".edf")[0] + ".csv_bi")


    X_dev = []
    y_raw_dev = []
    for eegpath, annpath, biannpath in \
            tqdm.tqdm(zip(subset_eegpaths_dev, subset_annpaths_dev, subset_binannpaths_dev),
                      total=len(subset_eegpaths_dev)):
        X, y_raw = munging.preprocess(eegpath, annpath, biannpath)
        
        X_dev.append(X.copy())
        y_raw_dev.append(y_raw.copy())

        del X
        del y_raw
        gc.collect()

    with open("X_dev.npy", "wb") as f:
        np.save(f, np.concatenate(X_dev))

    with open("y_dev.npy", "wb") as f:
        np.save(f, np.concatenate(y_raw_dev))

y_dev = munging.format_y_for_torcheeg(y_raw_dev)

# initialize torcheeg dataset
dev_set = NumpyDataset(X=X_dev,
                       y=y_dev,
                       io_path="./io/tuh_dev",
                       online_transform=transforms.ToTensor(),
                       label_transform=transforms.Compose([
                          transforms.Select('sz_present'),
                          transforms.Binary(2.0)]),
                       num_worker=1)