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

class CustomDataset(Dataset):
    def __init__(self, X, y, sz_thresh):
        self.X = zscore(X, axis=0).astype(np.float32)
        self.y = np.array(y >= sz_thresh).astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def plot_data(X, y):
    for idx,win in enumerate(X[:5]):
        plt.figure()
        plt.title(y[idx])
        plt.plot(win[0], alpha=0.3)
        plt.plot(win[1], alpha=0.3)
        plt.plot(win[2], alpha=0.3)
        plt.plot(win[3], alpha=0.3)
        plt.plot(win[4], alpha=0.3)


file_suffix = "_trn_blah"
sz_thresh = 2.0

if "_trn" in file_suffix:
	PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR
elif "_dev" in file_suffix:
	PATH_TO_EEGS = environ.TUH_DEV_DATA_DIR


eegpaths = glob.glob(os.path.join(PATH_TO_EEGS, "*/*/*/*.edf"))

subset_eegpaths = np.sort(eegpaths)
subset_annpaths = []
subset_binannpaths = []
for f in subset_eegpaths:
    subset_annpaths.append(f.split(".edf")[0] + ".csv")
    subset_binannpaths.append(f.split(".edf")[0] + ".csv_bi")

sz_types = []
for eegpath, annpath, biannpath in tqdm.tqdm(zip(subset_eegpaths, 
							        			 subset_annpaths, 
							        			 subset_binannpaths),
                  							 total=len(subset_eegpaths)):
	df_ann = pd.read_csv(annpath, comment="#")
	for _, r in df_ann.iterrows():
		sz_types.append(r.label)

import matplotlib.pyplot as plt 
import seaborn as sns 

df_analysis = pd.DataFrame(data={"ann": sz_types})
import IPython; IPython.embed()
