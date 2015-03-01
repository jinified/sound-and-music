from sys import argv
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import arff

script, filename = argv

N = 1024

def main():
	txt = open(filename)
	fileList = txt.readlines()
	txt.close()
	num_files = len(fileList)
	
	writeHeader()
	bufferMatrix = np.zeros((1290, N))
	dftMatrix = np.zeros((1290, N/2+1))
	
	for i in range(num_files):
		j, k = fileList[i].split("\t") #split string after \t
		rate, sample = wavfile.read(j) #read in wavfile
		sample = sample / 32768.0 #convert sample to floats
		sampleArray = np.array(sample) #creates array of sample values
		length = np.size(sampleArray) #number of samples of all files
	
		buffer_data = []
		num_buffer = int(length/N) * 2
		start = 0
		end = 1024
		for j in range(num_buffer):
			buffer_data = sampleArray[start:end]
			buffer_data = buffer_data * signal.hamming(N)
			bufferDFT = fft(buffer_data)
			bufferDFT = bufferDFT[:N/2+1]
			bufferDFT = np.abs(bufferDFT)
			dftMatrix[j,: ] = bufferDFT
			start = start + 512
			end = end + 512
		
		featureMatrix = np.zeros((num_buffer, 5))
		featureMatrix[:,0] = calcSC(dftMatrix)
		featureMatrix[:,1] = np.apply_along_axis(calcSRO, 1, dftMatrix)
		featureMatrix[:,2] = calcSFM(dftMatrix)
		featureMatrix[:,3] = calcPARFFT(dftMatrix)
		featureMatrix[:,4] = calcFLUX(dftMatrix)
	
		writeData(featureMatrix, k)

def writeHeader():
	f = open("dft.arff", "w")
	f.write('''@RELATION music_speech
@ATTRIBUTE SC_MEAN NUMERIC
@ATTRIBUTE SRO_MEAN NUMERIC
@ATTRIBUTE SFM_MEAN NUMERIC
@ATTRIBUTE PARFFT_MEAN NUMERIC
@ATTRIBUTE FLUX_MEAN NUMERIC
@ATTRIBUTE SC_STD NUMERIC
@ATTRIBUTE SRO_STD NUMERIC
@ATTRIBUTE SFM_STD NUMERIC
@ATTRIBUTE PARFFT_STD NUMERIC
@ATTRIBUTE FLUX_STD NUMERIC
@ATTRIBUTE class {music,speech}\n
@DATA\n''')
		
def calcSC(matrix):
	SCMatrix = np.copy(matrix)
	grid = np.indices((1290, 513))
	SCa = np.sum(grid[1]*SCMatrix, axis=1)
	SCb = np.sum(SCMatrix, axis=1)
	SC = SCa/SCb
	return SC
	
def calcSRO(matrix):
	SROMatrix = np.copy(matrix)
	SROcompare = 0.85*np.sum(SROMatrix)
	SROsum = 0
	
	for i in range(0, len(SROMatrix)):
		SROsum += SROMatrix[i]
		if SROsum >= SROcompare:
			return i
	
def calcSFM(matrix):
	gMean = np.exp(np.mean(np.log(matrix), axis=1))
	aMean = np.mean(matrix, axis=1)
	SFM = gMean/aMean
	return SFM

def calcPARFFT(matrix):
	RMS = np.sqrt(np.mean(np.square(matrix), axis=1))
	PAR = np.amax(matrix, axis = 1)
	PARFFT = PAR/RMS
	return PARFFT

def calcFLUX(matrix):
	SFMatrix = np.copy(matrix)
	minusOne = np.zeros(matrix.shape[1])
	SFprev = np.vstack([minusOne, matrix[:-1]])	
	SFdiff = SFMatrix - SFprev
	SF = np.sum(SFdiff.clip(0), axis=1)
	return SF
	
def writeData(featureMatrix, k):
	f = open("dft.arff", "a")
	writeMatrix = np.zeros(10)
	writeMatrix[0:5] = np.mean(featureMatrix, axis=0)
	writeMatrix[5:10] = np.std(featureMatrix, axis=0)
	SC_MEAN = writeMatrix[0]
	SRO_MEAN = writeMatrix[1]
	SFM_MEAN = writeMatrix[2]
	PARFFT_MEAN = writeMatrix[3]
	FLUX_MEAN = writeMatrix[4]
	
	SC_STD = writeMatrix[5]
	SRO_STD = writeMatrix[6]
	SFM_STD = writeMatrix[7]
	PARFFT_STD = writeMatrix[8]
	FLUX_STD = writeMatrix[9]
	
	f.write("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s" %(SC_MEAN, SRO_MEAN, SFM_MEAN, PARFFT_MEAN, FLUX_MEAN, SC_STD, SRO_STD, SFM_STD, PARFFT_STD, FLUX_STD, k))
	
main()