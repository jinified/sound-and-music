from sys import argv
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack
import numpy as np
import arff
import matplotlib.pyplot as plt

script, filename = argv

N = 1024
num_filters = 26

def main():
	txt = open(filename)
	fileList = txt.readlines()
	txt.close()
	num_files = len(fileList)
	writeHeader()
	
	for i in range(num_files):
		j, k = fileList[i].split("\t") #split string after \t
		rate, sample = wavfile.read(j) #read in wavfile
		sample = sample / 32768.0 #convert sample to floats
		sampleArray = np.array(sample) #creates array of sample values
		length = np.size(sampleArray) #number of samples of all files
		
		buffer_data = []
		num_buffer = int(length/1024) * 2
		buffer_matrix = np.zeros((num_buffer, N))
		start = 0
		end = 1024
		for x in range(num_buffer):
			buffer_data = sampleArray[start:end]
			start = start + 512
			end = end + 512
			buffer_matrix[x,: ] = buffer_data
		
		buffer_emp = preEmphasis(buffer_matrix) #apply pre-emphasis filter
		buffer_emp = buffer_emp * signal.hamming(N) #apply hamming window
		buffer_mag = magSpectrum(buffer_emp) #apply mag-spectrum
		
		melScale = calMelScale(rate) #calculate mel-scale
		filters = np.zeros((num_filters, 513))
		for y in range(num_filters):
			left, top, right = calFilters(y, melScale, rate/1024.0)
			filters[y] = triangulate(left, top, right)
		
		result = calMFCC(buffer_mag, filters.T)
		writeData(result, k)
		
	plot(filters, rate/2.0)
	
def preEmphasis (data):
	zeroMatrix = np.zeros((1290, N))
	temp = data[:, :-1]
	zeroMatrix[:, 1:N] = temp
	result = data - (0.95 * zeroMatrix)
	return result

def magSpectrum (buffer):
	fft = scipy.fftpack.fft(buffer, axis = 1)
	mag = np.abs(fft[:, 0:N / 2 + 1])
	return mag
	
def calMelScale(rate):
	mel = 1127 * np.log(1 + (rate / 2.0) / 700) / (num_filters + 1)
	return mel

def calFilters(y, mel, normalizer):
	left = ((np.exp((y * mel)/1127.0) - 1) * 700)/normalizer
	top = ((np.exp(((y + 1) * mel)/1127.0) - 1) * 700)/normalizer
	right = ((np.exp(((y + 2) * mel)/1127.0) - 1) * 700)/normalizer
	return left, top, right

def triangulate(left, top, right):
	left = np.floor(left)
	top = round(top)
	right = np.ceil(right)
	leftRange = top - left
	rightRange = right - top
	leftLine = np.linspace(0, 1, num = (leftRange + 1))
	leftLine = np.delete(leftLine, leftRange)
	rightLine = np.linspace(1, 0, num = (rightRange + 1))
	window = np.concatenate((np.zeros(left), leftLine, rightLine))
	result = np.concatenate((window, np.zeros(512 - right)))
	return result

def calMFCC(buffer, filters):
	result = np.dot(buffer, filters)
	result = np.log10(result)
	result = scipy.fftpack.dct(result)
	return result
	
def plot(windows, max):
	freq = np.linspace(0, max, num = 513)
	plt.figure()
	for i in range(num_filters):
		plt.plot(freq, windows[i])
	plt.ylabel("Amplitude")
	plt.xlabel("Frequency (Hz)")
	plt.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
	plt.savefig("figure1.png")

	plt.figure()
	for i in range(num_filters):
		plt.plot(freq, windows[i], '.-')
	plt.xlim(0, 300)
	plt.ylabel("Amplitude")
	plt.xlabel("Frequency (Hz)")
	plt.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
	plt.savefig("figure2.png")

def writeHeader():
	f = open("mfcc.arff", "w")
	f.write('@RELATION music_speech\n')
	for i in range(52):
		f.write('@ATTRIBUTE MFCC_%d NUMERIC\n' % (i))
	f.write('''@ATTRIBUTE class {music,speech}\n
@DATA\n''')
	
def writeData(result, k):
	f = open("mfcc.arff", "a")
	MeanMFCC = np.mean(result, axis = 0)
	StdMFCC = np.std(result, axis = 0)
	
	MEANstring = np.char.mod('%f', MeanMFCC)
	MEANstring = ",".join(MEANstring)
	STDstring = np.char.mod('%f', StdMFCC)
	STDstring = ",".join(STDstring)
	
	f.write("%s,%s,%s" %(MEANstring, STDstring, k))
	
main()