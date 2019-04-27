"""
Code based on the DeepSent project by Mu Chen as referenced in the final report https://github.com/muchen2/DeepSent
"""
import os
import pickle
import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft, dct
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tqdm import tqdm

import config


with open("models/arousal_regressor.pickle", "r") as file:
	arousal_regressor = pickle.load(file)
with open("models/valence_regressor.pickle", "r") as file:
	valence_regressor = pickle.load(file)


allowed_extensions = ["wav", "mp3", "ogg", "flac", "wma", "aac", "aiff", "m4a"]
wave_data_em = None
wave_data_gn = None
sound_filename = None

def analyse(filename, name):

	fn_parts = filename.split(".")
	extension = fn_parts[-1]
	if extension not in allowed_extensions:
		print("Skipping file as extension - " + extension + " - is invalid.")
		return None

	global wave_data_em, wave_data_gn, sound_filename
	aud = AudioSegment.from_file(filename, extension)
	aud_em = compress_audio_segment(aud, 11025, 1)
	aud_gn = compress_audio_segment(aud, 22050, 1)
	wave_data_em = np.asarray(aud_em.get_array_of_samples())
	wave_data_gn = np.asarray(aud_gn.get_array_of_samples())

	# Convert the middle 70% of the music into MFCC arrays which will be fed into the regressors
	frame_length = 5000
	frame_step = 500
	mfcc_frame_length = 25
	num_mfcc_coef_kept = 12
	frame_length_i = int(frame_length / 1000.0 * 11025) # compressed format always have sample rate of 11025
	frame_step_i = int(frame_step / 1000.0 * 11025)

	first = int(len(wave_data_em) * 0.15)
	last = int(len(wave_data_em) * 0.85)

	if (last - first) / 11025.0 < 5.0:
		# If the middle section is less than 5 seconds then the music is too short for analysis
		raise Error("The music you uploaded is too short. A length of at least 50/7 seconds is required")

	mid_segment = wave_data_em[first:last]
	num_frame = (len(mid_segment) - frame_length_i) // frame_step_i
	mfccs_mat = np.zeros((num_frame, int(frame_length / mfcc_frame_length * num_mfcc_coef_kept)))
	
	for i in range (num_frame):
		start_pos = i * frame_step_i
		end_pos = start_pos + frame_length_i
		mfccs = get_mfccs (mid_segment[start_pos:end_pos], sample_rate=11025, frame_length=mfcc_frame_length,
			frame_step=mfcc_frame_length, num_coef_kept=num_mfcc_coef_kept)
		mfccs_mat[i] = mfccs.flatten()

	# feed the data into regressors
	arousal_regressor_result = arousal_regressor.predict(mfccs_mat) + 1.5
	valence_regressor_result = valence_regressor.predict(mfccs_mat) + 1.5

	# shrink overrated scores
	arousal_regressor_result[np.where(arousal_regressor_result > 3.0)[0]] = 3.0
	arousal_regressor_result[np.where(arousal_regressor_result < 0.0)[0]] = 0.0
	valence_regressor_result[np.where(valence_regressor_result > 3.0)[0]] = 3.0
	valence_regressor_result[np.where(valence_regressor_result < 0.0)[0]] = 0.0

	# calculate mean results
	arousal_score = np.mean(arousal_regressor_result)
	valence_score = np.mean(valence_regressor_result)

	# calculate result ratios
	arousal_intense_ratio = len(np.where(arousal_regressor_result > 2.0)[0]) / float(len(arousal_regressor_result))
	arousal_relaxing_ratio = len(np.where(arousal_regressor_result < 1.0)[0]) / float(len(arousal_regressor_result))
	arousal_mid_ratio = 1.0 - arousal_intense_ratio - arousal_relaxing_ratio

	valence_happy_ratio = len(np.where(valence_regressor_result > 2.0)[0]) / float(len(valence_regressor_result))
	valence_sad_ratio = len(np.where(valence_regressor_result < 1.0)[0]) / float(len(valence_regressor_result))
	valence_neutral_ratio = 1.0 - valence_happy_ratio - valence_sad_ratio

	result_dict = {
		"location": filename, 
		"arousal_score": arousal_score / 3.0 * 100.0,
		"valence_score": valence_score / 3.0 * 100.0,
		"arousal_intense_ratio": arousal_intense_ratio * 100.0,
		"arousal_relaxing_ratio": arousal_relaxing_ratio * 100.0,
		"arousal_mid_ratio": arousal_mid_ratio * 100.0,
		"valence_happy_ratio": valence_happy_ratio * 100.0,
		"valence_sad_ratio": valence_sad_ratio * 100.0,
		"valence_neutral_ratio": valence_neutral_ratio * 100.0
	}
	print(result_dict)
	# np.save(os.path.join(config.MOOD_DIR, name), result_dict)


"""
Compress user input's audio segment by
	1. reducing the number of channels, if possible
	2. reducing the sample rate, if possible
"""
def compress_audio_segment(segment, sample_rate, num_channels):

	# limit the number of channels to be one
	if segment.channels > num_channels:
		segment_out = segment.set_channels (num_channels)

	# limit the segment's frame rate to be 11025
	if segment.frame_rate > sample_rate:
		segment_out = segment_out.set_frame_rate (sample_rate)

	return segment_out

"""
Generate hamming window function with specific length for FFT's usage
"""
def gen_hamming_window(length):
	alpha = 0.54
	beta = 1 - alpha
	return -beta * np.cos(np.arange(length) * 2 * np.pi / (length - 1)) + alpha

"""
Convert frequency (in hertz) to Mel scale
"""
def toMel(freq):
	return 1125 * np.log(1 + freq / 700.0)

"""
Convert Mel-scale frequency to Hertz frequency
"""
def toHertz(mel):
	return 700 * (np.exp(mel/1125) - 1)

"""
Generate Mel filterbank that convert the linear frequency resolution obtained in the result of FFT into nonlinear frequency resolution

Filterbank analysis is a process that simulates how human ears actually perceive sounds (http://www.ee.columbia.edu/ln/rosa/doc/HTKBook21/node54.html)

lower and upper are the lower and upper frequencies in hertz

n: number of filterbanks we want to generate
"""
def gen_mel_filters (nfft, sample_rate, n = 26, lower = 300, upper = 8000):
	lower_mel = toMel(lower)
	upper_mel = toMel(upper)
	mel_linspace = np.linspace(lower_mel, upper_mel, n + 2)
	hertz_linspace = toHertz(mel_linspace)

	# round the frequencies to nearest FFT bin to match
	# the frequency resolution
	fft_linspace = np.asarray(np.floor (nfft * hertz_linspace / sample_rate), dtype=np.int32)
	filters = []
	for i in range (n):
		filter = np.zeros(nfft)
		filter[fft_linspace[i]:fft_linspace[i+1] + 1] = np.linspace (0.0, 1.0, fft_linspace[i+1] - fft_linspace[i] + 1)
		filter[fft_linspace[i+1] + 1:fft_linspace[i+2] + 1] = np.linspace (1.0, 0.0, fft_linspace[i+2] - fft_linspace[i+1] + 1)[1:]
		filters.append(filter)

	return filters

"""
Extract MFC coefficients from the given signal data x

Implementation steps:
	1. Take DFT of windowed signal
	2. Calculate the Mel-scaled filterbank energy from the power spectrum obtained in step 1
	3. Take the log of each of the filterbank energy
	4. Take the DCT of the log filterbank energy to obtain the cepstrum coefficients
	(http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) 

frame_length and frame_step are in milliseconds

num_coef_kept: number of cepstral coefficients that will be kept in the final step of calculation
nfft: number of fft points
n_filters: number of mel filterbanks that will be used
frame_limit: if greater than zero, only this amount of frames will be processed in x

lower and upper are used as the same lower and upper in gen_mel_filter function declared above
"""
def get_mfccs(x, sample_rate, frame_length = 25, frame_step = 10, num_coef_kept = 13, n_filters = 26, frame_limit = 0, lower = 300, upper = 8000):

	# convert time length of frame length and frame step into
	# array length
	fsize = int(frame_length / 1000.0 * sample_rate)
	fstep = int(frame_step / 1000.0 * sample_rate)

	if frame_limit > 0:
		num_frame = frame_limit
	else:
		# frame limit is not set, process the whole data
		num_frame = (len(x) - fsize) // fstep + 1
	mfccs = np.zeros((num_frame, num_coef_kept))

	# construct hamming window and mel filterbank beforehand
	hamming_win = gen_hamming_window(fsize)
	mel_filters = gen_mel_filters(fsize, sample_rate, n = n_filters, lower = lower, upper = upper)

	# iterate through each frame and extract its mfcc
	for i in range(num_frame):

		# padding zeros at the end if the last frame goes out of bound
		if i * fstep + fsize > len(x):
			frame = np.zeros(fsize)
			rest_len = len(x) - i * fstep
			frame[:rest_len] = x[i*fstep:]
		else:
			frame = x[i * fstep:i * fstep + fsize]

		# calculate windowed DFT
		f_spectrum = fft(x = hamming_win * frame)

		# convert the complex numbers in the spectrum into their squared magnitudes (power)
		p_spectrum = np.absolute(f_spectrum) ** 2

		# filter the spectrum using mel filterbanks
		mel_energies = np.zeros(len(mel_filters))
		
		for j in range(len(mel_filters)):
			mel_energies[j] = np.dot(p_spectrum, mel_filters[j])

		# log the energies
		log_mel_energies = np.log(mel_energies + 1e-5)

		# compute DCT of the log energies
		mfcc = dct(log_mel_energies)
		mfccs[i, :] = mfcc[:num_coef_kept]

	return mfccs

if __name__ == '__main__':
	for filename in tqdm(os.listdir(config.AUDIO_DIR)):
		end = len(filename.split('.')[-1]) + 1
		analyse(os.path.join(config.AUDIO_DIR, filename), filename[:-end])