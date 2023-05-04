import sys
import numpy as np
import pandas as pd 
import mne 
import math

import environ, montages, config

def _read_montage_from_text(path_to_montage):
	montage = []
	with open(path_to_montage) as f:
		lines = f.readlines()
	lines = [l for l in lines if l[0] != "#"]
	for l in lines:
		if "[Montage]" in l or l == "\n":
			continue
		else:
			_mon = l.split(": ")[1]
			if " -- " in _mon:
				e1 = _mon.split(" -- ")[0].strip()
				e2 = _mon.split(" -- ")[1].strip()
				montage.append((e1, e2))
			else:
				if "EEG EKG-LE" in _mon:
					montage.append(("EEG EKG-LE",))
	return montage


def _calculate_montage(X, orig_ch_names, montage):
	data_new = []
	for chs in montage:
		if len(chs) == 2:
			ch0, ch1 = chs 
			idx0 = orig_ch_names.index(ch0)
			idx1 = orig_ch_names.index(ch1)
			data_new.append(X[idx0] - X[idx1])
		else:
			idx = orig_ch_names.index(chs[0])
			data_new.append(X[idx])
	return np.array(data_new)


def prepare_montage(raw_edf):
	ch_names = raw_edf.info["ch_names"]

	# select proper montage
	if "EEG A1-REF" in ch_names:
		montage = montages.MONTAGE_TCP_AR
	elif "EEG A1-LE" in ch_names:
		montage = montages.MONTAGE_TCP_LE
	elif "EEG FP1-LE" in ch_names:
		montage = montages.MONTAGE_TCP_LE_A
	elif "EEG FP1-REF" in ch_names:
		montage = montages.MONTAGE_TCP_AR_A
	else:
		raise Exception("this montage is not implemented!", ch_names)

	# perform the actual montaging
	X = _calculate_montage(raw_edf.get_data(),
						ch_names,
						montage)
	return X, montage


def simplify_labels(recording_dur_secs, df_label, df_label_bin):
	"""
	the current formatting of the labels is that labels are done on a per-
	channel basis, which is great but for now is a little overkill. 

	returns binary array y of size n (where n is len of dataset in secs), with 
		y[i] = 1 if there is seizure activity at time i, or y[i] = 0 otherwise
	"""

	# i'm thinking that if a sz shows up... annotations may not run the full
	# length of the recording
	# if recording_dur_secs != np.max(df_label.stop_time.values):
	# 	print("recording duration and max label stop time do not match!")

	y = np.zeros(int(recording_dur_secs))
	if df_label_bin.label.iloc[0] == "bckg":
		return y
	else:
		for _,r in df_label.iterrows():
			if "sz" in r.label:
				y[int(r.start_time): int(r.stop_time + 0.5)] = 1
		return y


def _debug_plot(X_montaged, df_ann, X, y):
	import matplotlib.pyplot as plt
	import os
	ch = 1
	fig, axes = plt.subplots(4, sharex=True, figsize=(15,8))
	axes[0].set_title("after montaging before windowing")
	t = np.arange(X_montaged.shape[1]) / sfreq
	axes[0].plot(t, X_montaged[ch])
	for _, r in df_ann.iterrows():
		if "sz" in r.label:
			axes[1].axvspan(r.start_time, r.stop_time, alpha=0.2)
	axes[2].set_title("after windowing")
	for i in range(len(X)):
		axes[2].plot(np.arange(i*640, (i+1)*640) / 64,
					 X[i][ch])
	axes[3].plot(np.arange(len(y))*10, y)
	plt.savefig("./debug/%s.png" % (os.path.basename(annpath).split(".csv")[0]))
	plt.close()


def window_and_label_data(X, y, win_size_secs, sfreq=256.0):
	"""
	takes eeg readings X (hopefully filtered and montaged by this point) of 
		varying length l (in samples), and splits into array of shape 

		(n_windows, n_channels, samps_per_win) 

		where 
			
		n_windows = math.floor(l / samps_per_win) and 
		samps_per_win = win_size_secs * sfreq

	takes per-second labels y and returns 1d vector array of len n_windows where
		y[i] = the number of seconds within the window that have sz activity
	"""
	n_channels, l = X.shape
	n_channels, l = X.shape
	samps_per_win = int(win_size_secs * sfreq)
	n_windows = int(math.floor(l / samps_per_win))

	X_new = []
	for i in range(n_windows):
		X_new.append(X[:, i*samps_per_win : (i+1)*samps_per_win])
	X_new = np.array(X_new)
	
	y_windowed = []
	for i in range(n_windows):
		y_windowed.append(y[int(i*win_size_secs) : int((i+1)*win_size_secs)])

	y_new = np.array([np.sum(y_win) for y_win in y_windowed]).astype(int)

	return X_new, y_new


def preprocess(eegpath, annpath, biannpath):
	"""
	Taking the eeg, annotation, binary annotation files for a single set, 
	turn them into a windowed X and y that can be concatenated together and 
	prepared for use with TorchEEG NumpyDataset.

	eegpath: file path to relevant edf file
	annpath: file path to annotation csv file
	biannpath: file path to binary csv-bi file

	Uses configuration values from config.py to determine filter cutoffs etc

	returns: X, y
		X is (n_windows, n_channels, samps_per_win) 
		y is 1d vector of len n_windows, where y[i] = # of seconds with sz
			activity for the window found at index i
	"""

	raw_edf = mne.io.read_raw_edf(eegpath, preload=True, verbose="ERROR")
	df_ann = pd.read_csv(annpath, comment="#")
	df_biann = pd.read_csv(biannpath, comment="#")

	# filter
	raw_edf.filter(config.LF, config.HF, fir_design='firwin', verbose='ERROR')

	# resample
	if raw_edf.info['sfreq'] != config.SFREQ:
		raw_edf = raw_edf.copy().resample(config.SFREQ)
		
	# get some metadata, AFTER resampling
	sfreq = raw_edf.info['sfreq']
	duration_secs = raw_edf.get_data().shape[1] / sfreq

	# montage
	X_montaged, montage = prepare_montage(raw_edf)

	# simplify labels
	y = simplify_labels(duration_secs, 
						df_ann,
						df_biann)

	# window data and labels
	X, y = window_and_label_data(X_montaged, y, config.WIN_SIZE_SECS, config.SFREQ)
	return X, y


def format_y_for_torcheeg(y):
	"""
	Given y where y is 1d vector of len n, where: 
		n = (duration of dataset in seconds) / (secs_in_win)
		y[i] = # of seconds w/ sz activity in the window at index i

	returns dict Y that will play nice with torcheeg NumpyDataset
	"""
	sz_present_arr = []
	sz_absent_arr = []
	for y_i in y:
		sz_present_arr.append(int(np.sum(y_i)))
		sz_absent_arr.append(int(config.WIN_SIZE_SECS - np.sum(y_i)))
	Y = {"sz_present": np.array(sz_present_arr),
		 "sz_absent": np.array(sz_absent_arr)}

	return Y




