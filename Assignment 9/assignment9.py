#Author: Omar Khalid bin Yahya
#Matric. No: A0094534B

from scipy.io.wavfile import read, write
from scipy import signal
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt

infile = "clear_d1.wav"
N = 128 #window size
fs = 22050.0

def main():	
	freq_amp, num_buffer, db_orig = analyze()
	reconstruct(freq_amp, num_buffer)
	spectrogram(num_buffer, db_orig)

def analyze():
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
	return freq_amp, num_buffer, db_data

def reconstruct(freq_amp, num_buffer):
	re_wav = np.zeros(0)
	
	for x in range(num_buffer):
		wav = calcSin(freq_amp[x, 0], freq_amp[x, 1])
		re_wav = np.concatenate((re_wav, wav))
	
	re_wav_norm = re_wav / re_wav.max() * 32767
	re_wav_norm = re_wav_norm.astype(np.int16)
	write("reconstructed.wav", fs, re_wav_norm)

def calcSin(freq, amp):
	wav = amp * np.sin(2 * np.pi * freq / fs * np.arange(N))
	return wav
	
def spectrogram(num_buffer, db_orig):
	rate, sample = read("reconstructed.wav") #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	buffer_new = np.zeros((num_buffer, N))
	
	for x in range(num_buffer):
		start = x * N
		end = start + N
		slice = sample[start:end]		
		buffer_new[x,:] = slice
	
	db_new = power_spectrum(buffer_new, np.hamming(N))
	plot(db_orig, db_new)
	
def plot(orig, new):
	plt.subplot(2, 1, 1)
	plt.title("Spectrogram")
	plt.imshow(orig.T / orig.max(), origin = "lower", aspect = "auto")
	plt.xlabel("frames (original wav)")
	plt.ylabel("freq bin")

	plt.subplot(2, 1, 2)
	plt.imshow(new.T / new.max(), origin = "lower", aspect = "auto")
	plt.xlabel("frames (reconstructed wave)")
	plt.ylabel("freq bin")
	
	plt.savefig("spectrogram.png")
	
def power_spectrum(x, window):
	fft = scipy.fftpack.fft(x * window, axis = 1)
	db = abs(fft[:,0:N/2+1])
	return db
	
main()