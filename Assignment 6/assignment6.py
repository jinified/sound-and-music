#Author: Omar Khalid bin Yahya
#Matric. No: A0094534B

from sys import argv
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack
import numpy as np

script, filename = argv

N = 1024

def main():	
	rate, sample = wavfile.read(filename) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	sampleArray = np.array(sample) #creates array of sample values
	length = np.size(sampleArray) #number of samples of all files
		
	buffer_data = []
	num_buffer = int(length/(N / 2) - 1)
	buffer_matrix = np.zeros((num_buffer, N))
	start = 0
	end = N
	for x in range(num_buffer):
		buffer_data = sampleArray[start:end]
		start = start + 512
		end = end + 512
		buffer_matrix[x,: ] = buffer_data

	buffer_matrix = buffer_matrix * signal.hamming(N) #apply hamming window
	buffer_mag = magSpectrum(buffer_matrix) #apply mag-spectrum
	accent_signal = calAccent(num_buffer, buffer_mag) #calculate accent signal
	auto_signal = autocorr(accent_signal)
	
	upper_index = int(np.ceil(60/(60*0.0116)))
	lower_index = int(np.floor(60/(180*0.0116)))
	tempo_index = getMax(auto_signal, lower_index, upper_index)
	result = beat_analysis(accent_signal, tempo_index)
	result = np.array(result) * 0.0116
	
	writeData(result)

def magSpectrum (buffer):
	fft = scipy.fftpack.fft(buffer, axis = 1)
	mag = np.abs(fft[:, 0:N / 2 + 1])
	return mag
	
def calAccent (num_buffer, buffer_mag):
	a = np.copy(buffer_mag)
	a = np.abs(a[0:num_buffer - 1])
	b = np.copy(buffer_mag)
	b = np.abs(b[1:num_buffer])
	
	result = b - a
	result = np.where(result < 0, 0, result)
	sum = np.sum(result, axis = 1)
	return sum

def autocorr(x):
	result = np.correlate(x, x, mode='full')
	return result[result.size / 2:]

def getMax(array, low, up):
	result = array[low:up+1].argmax()
	return result + low
	
def beat_analysis(accent_signal, t):
	a = np.copy(accent_signal)
	firstBeat = a[0:t].argmax()
	beats = []
	beats.append(firstBeat)
	count = firstBeat + t
	while count < len(accent_signal):
		low = count - 10
		up = count + 10
		beatIndex = getMax(accent_signal, low, up)
		beats.append(beatIndex)
		count = beatIndex
		count = count + t	
	return beats
	
def writeData(result):
	result.tofile('beat_time.csv', sep=',')
	
main()