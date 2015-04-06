#Author: Omar Khalid bin Yahya
#Matric. No: A0094534B

from scipy.io.wavfile import read, write
from scipy import signal
import scipy.fftpack
import numpy as np

import os
os.getcwd()

infile = "clear_d1.wav"
N = 128 #window size

def main():
	rate, sample = read(infile) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	num_buffer = len(sample) / N
	buffer_data = np.zeros((num_buffer, 65))
	freq_amp = np.zeros((num_buffer, 2))
	bin2freq = rate / N
	
	for x in range(num_buffer):
		start = int(x * 128)
		end = start + 128
		result = power_spectrum(sample[start:end], np.hamming(N))
		buffer_data[x,:] = result
	
	freq_amp[:,0] = np.argmax(buffer_data, axis = 1) * bin2freq #frequency
	freq_amp[:,1] = np.amax(buffer_data, axis = 1) #amplitude
	
	np.savetxt('freq_amp.csv', freq_amp, fmt='%.6g', delimiter=',')

def power_spectrum(x, window):
	fft = np.fft.fft(x * window)
	# only keep the positive frequencies
	fft = fft[:len(fft) / 2 + 1]
	# magnitude spectrum, normalize
	magfft = abs(fft) / (np.sum(window) / 2.0)
	# log-spectrum
	epsilon = 1e-10
	db = 20 * np.log10(magfft + epsilon)
	return db
	
main()