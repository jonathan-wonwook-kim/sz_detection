import numpy as np
import torch
from torch import nn
import glob, os

import environ
import config

from munging import preprocess
from models import *

def szdetect_events(edf_path):
	"""
	Given an edf filepath, load the edf, filter, montage, and window it, then
	call a model on the windowed eeg signal to detect seizures.
	"""
	win_size = config.WIN_SIZE_SECS

	device = "mps"
	X, _ = preprocess(edf_path)

	model = CNN(in_channels = 20)
	model.load_state_dict(torch.load("/Users/jonathan/Dev/Research/Bernardo/eeggpt/seizure_detection/cnn1.model"))
	model.eval()
	# print(summary(model, input_size=(64, 20, 640)))
	model.to(device)

	sig = nn.Sigmoid()

	X = torch.Tensor(X).to(device)

	pred = model(X)
	preds_bi = sig(pred).detach().cpu().numpy()

	return preds_bi, win_size


def test():
	# find load paths
	PATH_TO_EEGS = environ.TUH_TRAIN_DATA_DIR

	subj_dirs = np.sort(glob.glob(os.path.join(PATH_TO_EEGS, "*")))
	subj_dirs = subj_dirs[1::2]

	# get per-subject edf paths
	eegpaths_by_subj = [glob.glob(os.path.join(s, "*/*/*.edf")) \
						for s in subj_dirs]

	# flatten list of edf paths and get their stems
	eegpaths = [item for sublist in eegpaths_by_subj for item in sublist]

	preds_bi, win_size = detect_seizures(eegpaths[0])
	import IPython; IPython.embed()



if __name__ == '__main__':
	test()
