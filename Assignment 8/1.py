#Author: Omar Khalid bin Yahya
#Matric. No: A0094534B

from scipy.io.wavfile import write
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

fs = 44100.0
freq = 1000.0
duration = 1.0
amplitude = 0.5

def main():
	maxSine = int(np.floor(fs / 2 / freq))
	constructed = np.zeros((maxSine, duration * fs))
	
	for i in range(1, maxSine + 1):
		temp = sin(freq, i, duration)
		constructed[i - 1] = temp
		
	result = sawtooth(constructed)
	
	t = np.arange(int(duration * fs))
	perfect = amplitude * scipy.signal.sawtooth(freq / fs * 2.0 * np.pi * t)
	
	plotTimeDomain(perfect, result)
	plotDBMagFFT(power_spectrum(perfect[0:8192]), power_spectrum(result[0:8192]))
	
	write("sawtooth_constructed.wav", fs, result)
	write("sawtooth_perfect.wav", fs, perfect)

def sin(freq, k, duration):
	period = np.arange(int(duration * fs))
	wave = pow(k, -1) * np.sin(k * 2.0 * np.pi * (freq / fs) * period)
	return wave
	
def sawtooth(array):
	total = np.sum(array, axis = 0)
	result = ((-2 * amplitude) / np.pi) * total
	return result

def power_spectrum(x):
	window = np.blackman(len(x))
	fft = np.fft.fft(x * window)
	# only keep the positive frequencies
	fft = fft[:len(fft) / 2 + 1]
	# magnitude spectrum, normalize
	magfft = abs(fft) / (np.sum(window) / 2.0)
	# log-spectrum
	epsilon = 1e-10
	db = 20 * np.log10(magfft + epsilon)
	return db

def plotTimeDomain(data1, data2):
	#data1 = perfect sawtooth
	#data2 = reconstructed
	plt.xlabel("Samples")
	plt.ylabel("Amplitude")
	plot1 = plt.plot(data1, c = "blue", label = "perfect sawtooth")
	plot2 = plt.plot(data2, c = "green", label = "reconstructed")
	plt.legend(loc = 1)
	plt.xlim(0, 240)
	plt.savefig('Time Domain.png')
	
def plotDBMagFFT(data1, data2):
	plt.clf()
	plt.xlabel("FFT bin")
	plt.ylabel("dB")
	plot1 = plt.plot(data1, c = "blue", label = "perfect sawtooth")
	plot2 = plt.plot(data2, c = "green", label = "reconstructed")
	plt.legend(loc = 1)
	plt.xlim(0, 4096)
	plt.savefig('DB-Mag FFT.png')

main()