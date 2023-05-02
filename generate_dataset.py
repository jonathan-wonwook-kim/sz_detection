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

file_suffix = "_trn_xsm"

if "_trn" in file_suffix:
	PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR
elif "_dev" in file_suffix:
	PATH_TO_EEGS = environ.TUH_DEV_DATA_DIR

if os.path.isfile("X%s.npy" % file_suffix) and os.path.isfile("y%s.npy" % file_suffix):
	with open("X%s.npy" % file_suffix, "rb") as f:
		X = np.load(f)
	with open("y%s.npy" % file_suffix, "rb") as f:
		y_raw = np.load(f)
else:
	eegpaths = glob.glob(os.path.join(PATH_TO_EEGS, "*/*/*/*.edf"))

	subset_eegpaths = np.sort(eegpaths)[:2]
	subset_annpaths = []
	subset_binannpaths = []
	for f in subset_eegpaths:
	    subset_annpaths.append(f.split(".edf")[0] + ".csv")
	    subset_binannpaths.append(f.split(".edf")[0] + ".csv_bi")

	X = []
	y_raw = []
	for eegpath, annpath, biannpath in tqdm.tqdm(zip(subset_eegpaths, 
								        			 subset_annpaths, 
								        			 subset_binannpaths),
	                  							 total=len(subset_eegpaths)):
	    X_curr, y_raw_curr = munging.preprocess(eegpath, annpath, biannpath)
	    
	    X.append(X_curr.copy())
	    y_raw.append(y_raw_curr.copy())

	    del X_curr; del y_raw_curr; gc.collect()

	X = np.concatenate(X)
	y_raw = np.concatenate(y_raw)

	with open("X%s.npy" % file_suffix, "wb") as f:
	    np.save(f, X)
	with open("y%s.npy" % file_suffix, "wb") as f:
	    np.save(f, y_raw)

y = munging.format_y_for_torcheeg(y_raw)

# initialize torcheeg dataset
dataset = NumpyDataset(X = X,
                       y = y,
                       io_path="./io/tuh%s" % file_suffix,
                       online_transform=transforms.Compose([
                         transforms.MeanStdNormalize(axis=0),
                         transforms.ToTensor()]),
                       label_transform=transforms.Compose([
                         transforms.Select('sz_present'),
                         transforms.Binary(sz_thresh)]),
                       num_worker=1)


# TODO split trn set into several NumpyDatasets
# TODO figure out why the resulting X_trn is so big compared to eeg size...