
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

def main(which_set, sub_set, wins_per_file=config.WINS_PER_CACHE_FILE, 
		 regen=False, sz_thresh=2.0):

	if which_set == "trn":
		PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR
	elif which_set == "dev":
		PATH_TO_EEGS = environ.TUH_DEV_DATA_DIR

	subj_dirs = np.sort(glob.glob(os.path.join(PATH_TO_EEGS, "*")))
	num_subjs = len(subj_dirs)

	# subselect subjects if you want
	if sub_set == "first_half":
		subj_dirs = subj_dirs[0::2] # take every other subj starting at ind 0
	elif sub_set == "second_half":
		subj_dirs = subj_dirs[1::2]
	elif sub_set == "all":
		subj_dirs = subj_dirs

	eegpaths_by_subj = [glob.glob(os.path.join(s, "*/*/*.edf")) \
						for s in subj_dirs]

	eegpaths = [item for sublist in eegpaths_by_subj for item in sublist]
	stems = [Path(eegpath).stem for eegpath in eegpaths]

	subset_eegpaths = np.sort(eegpaths) #[:4]
	subset_annpaths = []
	subset_biannpaths = []
	for f in subset_eegpaths:
		subset_annpaths.append(f.split(".edf")[0] + ".csv")
		subset_biannpaths.append(f.split(".edf")[0] + ".csv_bi")

	batch_dx = 0
	X_batch = []
	y_batch = []
	for eegpath, annpath, biannpath in tqdm.tqdm(zip(subset_eegpaths, 
													 subset_annpaths, 
													 subset_biannpaths),
												 total=len(subset_eegpaths)):
		X_curr, y_raw_curr = munging.preprocess(eegpath, annpath, biannpath)
		if X_curr.shape[0] > 0:

			X_batch.append(X_curr)
			y_batch.append(y_raw_curr)

			if np.concatenate(X_batch).shape[0] > wins_per_file: 
				X_cum = np.concatenate(X_batch)
				y_cum = np.concatenate(y_batch)

				idx = 0
				while idx < (len(X_cum) - wins_per_file):
					ind_batch_st = idx
					ind_batch_en = idx + wins_per_file

					X_batch_save = X_cum[ind_batch_st:ind_batch_en]
					y_batch_save = y_cum[ind_batch_st:ind_batch_en]

					save_path_X_curr = "./cache/preprocessed/%s/%s/batch_%d_X.npy" % \
													(which_set, sub_set, batch_dx)
					save_path_y_curr = "./cache/preprocessed/%s/%s/batch_%d_y.npy" % \
													(which_set, sub_set, batch_dx)

					with open(save_path_X_curr, "wb") as f:
						np.save(f, X_batch_save)
					with open(save_path_y_curr, "wb") as f:
						np.save(f, y_batch_save)

					del X_batch_save
					del y_batch_save

					batch_dx += 1
					idx += wins_per_file

				X_batch = [X_cum[idx:]]
				y_batch = [y_cum[idx:]]
	print("\nsaved %d batch(es)" % (batch_dx))
	print("losing %d window(s)" % np.concatenate(X_batch).shape[0])
	assert np.concatenate(X_batch).shape[0] < wins_per_file

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-d", "--dataset", dest="dataset")
	parser.add_option("-s", "--subset", dest="subset")

	options, _ = parser.parse_args()

	main(options.dataset, options.subset, regen=True)


