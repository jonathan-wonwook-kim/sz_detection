
import os
import sys
import numpy as np
import mne
import pandas as pd
import tqdm
import glob
import gc
import argparse
import shutil
from optparse import OptionParser
from pathlib import Path

import torcheeg
from torcheeg.datasets import NumpyDataset
from torcheeg import transforms

import environ
import config
import munging

import matplotlib.pyplot as plt

def main(dataset, subset=None, size=None, regen=False, sz_thresh=2.0):
	# define save paths
	if subset is not None:
		save_path_X = "./cache/X_%s_%s.npy" % (dataset, subset)
		save_path_y = "./cache/y_%s_%s.npy" % (dataset, subset)
		cache_path = "./io/tuh_%s_%s" % (dataset, subset)
	print(cache_path)

	# find load paths
	if dataset == "trn":
		PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR
	elif dataset == "dev":
		PATH_TO_EEGS = environ.TUH_DEV_DATA_DIR
	elif dataset == "eval":
		PATH_TO_EEGS = environ.TUH_EVAL_DATA_DIR

	if not regen and \
			(os.path.isfile(save_path_X) and os.path.isfile(save_path_y)):
		with open(save_path_X, "rb") as f:
			X = np.load(f)
		with open(save_path_y, "rb") as f:
			y_raw = np.load(f)
	else:
		# get subject-level directories
		subj_dirs = np.sort(glob.glob(os.path.join(PATH_TO_EEGS, "*")))
		num_subjs = len(subj_dirs)

		# subselect subjects
		if subset == "first_half":
			subj_dirs = subj_dirs[0::2] # take every other subj starting at ind 0
		elif subset == "second_half":
			subj_dirs = subj_dirs[1::2]
		elif subset == "all":
			subj_dirs = subj_dirs

		# get per-subject edf paths
		eegpaths_by_subj = [glob.glob(os.path.join(s, "*/*/*.edf")) \
							for s in subj_dirs]

		# flatten list of edf paths and get their stems
		eegpaths = [item for sublist in eegpaths_by_subj for item in sublist]
		stems = [Path(eegpath).stem for eegpath in eegpaths]

		# if you want smaller dataset, take first 'size' paths
		if size is not None and type(size) == int:
			subset_eegpaths = np.sort(eegpaths)[:size]
		else:
			subset_eegpaths = np.sort(eegpaths)
		subset_annpaths = []
		subset_biannpaths = []
		for f in subset_eegpaths:
		    subset_annpaths.append(f.split(".edf")[0] + ".csv")
		    subset_biannpaths.append(f.split(".edf")[0] + ".csv_bi")

		X = []
		y_raw = []
		for eegpath, annpath, biannpath in tqdm.tqdm(zip(subset_eegpaths, 
									        			 subset_annpaths, 
									        			 subset_biannpaths),
		                  							 total=len(subset_eegpaths)):
		    X_curr, y_raw_curr = munging.preprocess(eegpath, annpath, biannpath)
		    if len(X_curr) == 0:
		    	print("%s has less than %d secs of data, skipping" % \
		    				(os.path.basename(eegpath), \
		    				 int(config.WIN_SIZE_SECS)))
		    else:
			    X.append(X_curr.copy())
			    y_raw.append(y_raw_curr.copy())

		    del X_curr; del y_raw_curr; gc.collect()

		X = np.concatenate(X)
		y_raw = np.concatenate(y_raw)

		with open(save_path_X, "wb") as f:
		    np.save(f, X)
		with open(save_path_y, "wb") as f:
		    np.save(f, y_raw)

    # initialize torcheeg dataset
	y = munging.format_y_for_torcheeg(y_raw)
	dataset = NumpyDataset(X = X,
	                       y = y,
	                       io_path=cache_path,
	                       io_mode="pickle", #"lmdb",
	                       online_transform=transforms.Compose([
	                         transforms.MeanStdNormalize(axis=1),
	                         transforms.ToTensor()]),
	                       label_transform=transforms.Compose([
	                         transforms.Select('sz_present'),
	                         transforms.Binary(sz_thresh)]),
	                       num_worker=32)

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-d", "--dataset", dest="dataset")
	parser.add_option("-s", "--subset", dest="subset")
	parser.add_option("-z", "--size", dest="size", default=None)

	options, _ = parser.parse_args()

	main(dataset=options.dataset, 
		 subset=options.subset, 
		 size=options.size, 
		 regen=False)

# TODO split trn set into several NumpyDatasets
# TODO figure out why the resulting X_trn is so big compared to eeg size...