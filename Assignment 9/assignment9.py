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
fs = 22050.0

def main():
	rate, sample = read(infile) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	num_buffer = len(sample) / N
	buffer_data = np.zeros((num_buffer, N))
	freq_amp = np.zeros((num_buffer, 2))
	
	for x in range(num_buffer):
		start = x * N
		end = start + N
		slice = sample[start:end]		
		buffer_data[x,:] = slice
	
	db_data = power_spectrum(buffer_data, np.hamming(N))	
	maxIndex = np.argmax(db_data, axis = 1)
	maxIndexFreq = maxIndex/float(N) * fs #frequency
	freq_amp[:,0] = maxIndexFreq
	freq_amp[:,1] = np.amax(db_data, axis = 1) #amplitude
	
	np.savetxt('freq_amp.csv', freq_amp, fmt='%.6g', delimiter=',')

def power_spectrum(x, window):
	fft = scipy.fftpack.fft(x * window, axis = 1)
	db = abs(fft[:,0:N/2+1])
	return db
	
main()