
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

import torcheeg
from torcheeg.datasets import NumpyDataset
from torcheeg import transforms

import environ
import config
import munging

import matplotlib.pyplot as plt

def main(file_suffix, regen=False, sz_thresh=2.0):
	save_path_X = "./cache/X%s.npy" % file_suffix
	save_path_y = "./cache/y%s.npy" % file_suffix
	cache_path = "./io/tuh%s" % file_suffix
	print(cache_path)

	if "_trn" in file_suffix:
		PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR
	elif "_dev" in file_suffix:
		PATH_TO_EEGS = environ.TUH_DEV_DATA_DIR

	if not regen and \
			(os.path.isfile(save_path_X) and os.path.isfile(save_path_y)):
		with open(save_path_X, "rb") as f:
			X = np.load(f)
		with open(save_path_y, "rb") as f:
			y_raw = np.load(f)
	else:
		# if os.path.isdir(cache_path) and regen:
		# 	shutil.rmtree(cache_path)

		eegpaths = glob.glob(os.path.join(PATH_TO_EEGS, "*/*/*/*.edf"))

		first_n_files = 200 if "_sm" in file_suffix else len(eegpaths)

		subset_eegpaths = np.sort(eegpaths)[:first_n_files]
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
	# y = munging.format_y_for_torcheeg(y_raw)
	# dataset = NumpyDataset(X = X,
	#                        y = y,
	#                        io_path=cache_path,
	#                        online_transform=transforms.Compose([
	#                          transforms.MeanStdNormalize(axis=1),
	#                          transforms.ToTensor()]),
	#                        label_transform=transforms.Compose([
	#                          transforms.Select('sz_present'),
	#                          transforms.Binary(sz_thresh)]),
	#                        num_worker=1)

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-d", "--dataset", dest="dataset")
	parser.add_option("-s", "--size", dest="size", default="")

	options, _ = parser.parse_args()

	file_suffix = "_%s%s" % (options.dataset, options.size)

	main(file_suffix, regen=True)

# TODO split trn set into several NumpyDatasets
# TODO figure out why the resulting X_trn is so big compared to eeg size...