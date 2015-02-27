from sys import argv
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import arff

script, filename = argv

def main():
	txt = open(filename)
	fileList = txt.readlines()
	txt.close()
	num_files = len(fileList)
	
	writeHeader()
	bufferMatrix = np.zeros((1290, 1024))
	dftMatrix = np.zeros((1290, 1024))
	
	for i in range(num_files):
		j, k = fileList[i].split("\t") #split string after \t
		rate, sample = wavfile.read(j) #read in wavfile
		sample = sample / 32768.0 #convert sample to floats
		sampleArray = np.array(sample) #creates array of sample values
		length = np.size(sampleArray) #number of samples of all files
	
		buffer_data = []
		num_buffer = int(length/1024) * 2
		start = 0
		end = 1024
		for j in range(num_buffer):
			buffer_data = sampleArray[start:end]
			buffer_data = buffer_data * signal.hamming(len(buffer_data))
			bufferDFT = fft(buffer_data)
			bufferDFT = np.array([x for x in bufferDFT if x >= 0])
			dftMatrix[j,: ] = bufferDFT
			start = start + 512
			end = end + 512
	
		#featureMatrix = np.zeros((num_buffer, 5))
		#featureMatrix[:,0] = calcRMS(bufferMatrix)
		#featureMatrix[:,1] = calcPAR(bufferMatrix, featureMatrix[:,0])
		#featureMatrix[:,2] = calcZCR(bufferMatrix)
		#featureMatrix[:,3] = calcMAD(bufferMatrix)
		#featureMatrix[:,4] = calcMEAN_AD(bufferMatrix)
	
		#writeData(featureMatrix, k)

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
		
def calcRMS(bufferMatrix):
	RMSMatrix = np.copy(bufferMatrix)
	RMSMatrix = np.power(RMSMatrix, 2)
	RMS = np.sum(RMSMatrix, axis = 1)
	RMS = RMS/1024
	RMS = np.sqrt(RMS)
	return RMS
	
def calcPAR(bufferMatrix, RMS):
	PARMatrix = np.copy(bufferMatrix)
	PARMatrix = np.absolute(PARMatrix)
	PAR = np.max(PARMatrix, axis = 1)
	PAR = PAR/RMS
	return PAR
	
def calcZCR(bufferMatrix):
	ZCRMatrix = np.copy(bufferMatrix)
	ZCRa = ZCRMatrix[:1290, :1023]
	ZCRb = ZCRMatrix[:1290, 1:1024]
	ZCRc = ZCRa * ZCRb
	ZCR = np.where(ZCRc < 0, 1, 0)
	ZCR = np.sum(ZCR, axis = 1)
	ZCR = ZCR/1023.0
	return ZCR

def calcMAD(bufferMatrix):
	MADMatrix = np.copy(bufferMatrix)
	firstMed = np.median(MADMatrix, axis = 1)
	firstMed = np.reshape(firstMed, (len(firstMed), 1))
	MADMatrix = np.absolute(MADMatrix - firstMed)
	MAD = np.median(MADMatrix, axis = 1)
	return MAD

def calcMEAN_AD(bufferMatrix):
	MEANMatrix = np.copy(bufferMatrix)
	firstMean = np.mean(MEANMatrix, axis = 1)
	firstMean = np.reshape(firstMean, (len(firstMean), 1))
	MEANMatrix = np.absolute(MEANMatrix - firstMean)
	MEAN_AD = np.mean(MEANMatrix, axis = 1)
	return MEAN_AD
	
def writeData(featureMatrix, k):
	f = open("dft.arff", "a")
	writeMatrix = np.zeros(10)
	writeMatrix[0:5] = np.mean(featureMatrix, axis = 0)
	writeMatrix[5:10] = np.std(featureMatrix, axis = 0)
	RMS_MEAN = writeMatrix[0]
	PAR_MEAN = writeMatrix[1]
	ZCR_MEAN = writeMatrix[2]
	MAD_MEAN = writeMatrix[3]
	MEAN_AD_MEAN = writeMatrix[4]
	
	RMS_STD = writeMatrix[5]
	PAR_STD = writeMatrix[6]
	ZCR_STD = writeMatrix[7]
	MAD_STD = writeMatrix[8]
	MEAN_AD_STD = writeMatrix[9]
	
	f.write("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s" %(RMS_MEAN, PAR_MEAN, ZCR_MEAN, MAD_MEAN, MEAN_AD_MEAN, RMS_STD, PAR_STD, ZCR_STD, MAD_STD, MEAN_AD_STD, k))
	
main()