from sys import argv
from scipy.io import wavfile
import numpy as np
import arff

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
num_files = len(fileList)
bufferMatrix = np.zeros((1290, 1024))

f = open("whole-song.arff", "w")
f.write('''@RELATION music_speech
@ATTRIBUTE RMS NUMERIC
@ATTRIBUTE PAR NUMERIC
@ATTRIBUTE ZCR NUMERIC
@ATTRIBUTE MAD NUMERIC
@ATTRIBUTE class {music,speech}\n
@DATA\n''')

g = open("buffer-based.arff", "w")
g.write('''@RELATION music_speech
@ATTRIBUTE RMS_MEAN NUMERIC
@ATTRIBUTE PAR_MEAN NUMERIC
@ATTRIBUTE ZCR_MEAN NUMERIC
@ATTRIBUTE MAD_MEAN NUMERIC
@ATTRIBUTE MEAN_AD_MEAN NUMERIC
@ATTRIBUTE RMS_STD NUMERIC
@ATTRIBUTE PAR_STD NUMERIC
@ATTRIBUTE ZCR_STD NUMERIC
@ATTRIBUTE MAD_STD NUMERIC
@ATTRIBUTE MEAN_AD_STD NUMERIC
@ATTRIBUTE class {music,speech}\n
@DATA\n''')

for i in range(num_files):
	j, k = fileList[i].split("\t") #split string after \t
	rate, sample = wavfile.read(j) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	sampleArray = np.array(sample) #creates array of sample values
	length = np.size(sampleArray) #number of samples of all files
	RMS = np.sqrt((1/ float(length))*(np.sum(np.square(sampleArray))))
	PAR = (np.max(np.absolute(sampleArray)))/RMS
	
	#to calculate ZCR
	npSign = np.sign(sampleArray)
	npBinary = npSign[1:] * npSign[:-1]
	npNeg = np.array([x for x in npBinary if x < 0])
	npNeg[npNeg < 0] = 1
	ZCR = (1/float(length-1))*(np.sum(npNeg))

	MAD = np.median(np.abs(sampleArray - np.median(sampleArray)))
	f.write("%f,%f,%f,%f,%s" %(RMS, PAR, ZCR, MAD, k)) #write to original arff file

	buffer_data = []
	num_buffer = int(length/1024) * 2
	start = 0
	end = 1024
	for j in range(num_buffer):
		buffer_data = sampleArray[start:end]
		bufferMatrix[j,: ] = buffer_data
		start = start + 512
		end = end + 512
		
	featureMatrix = np.zeros((num_buffer, 5))
	
	#to calculate RMS
	#RMS = np.sqrt(np.mean(np.square(bufferMatrix)))
	RMSMatrix = np.copy(bufferMatrix)
	RMSMatrix = np.power(RMSMatrix, 2)
	RMS = np.sum(RMSMatrix, axis = 1)
	RMS = RMS/1024
	RMS = np.sqrt(RMS)
	
	#to calculate PAR
	#PAR = (np.max(np.absolute(bufferMatrix)))/RMS
	PARMatrix = np.copy(bufferMatrix)
	PARMatrix = np.absolute(PARMatrix)
	PAR = np.max(PARMatrix, axis = 1)
	PAR = PAR/RMS
		
	#to calculate ZCR
	ZCRMatrix = np.copy(bufferMatrix)
	ZCRa = ZCRMatrix[:1290, :1023]
	ZCRb = ZCRMatrix[:1290, 1:1024]
	ZCRc = ZCRa * ZCRb
	ZCR = np.where(ZCRc < 0, 1, 0)
	ZCR = np.sum(ZCR, axis = 1)
	ZCR = ZCR/1023.0

	#to calculate MAD
	#MAD = np.median(np.abs(buffer_data - np.median(bufferMatrix)))
	MADMatrix = np.copy(bufferMatrix)
	firstMed = np.median(MADMatrix, axis = 1)
	firstMed = np.reshape(firstMed, (len(firstMed), 1))
	MADMatrix = np.absolute(MADMatrix - firstMed)
	MAD = np.median(MADMatrix, axis = 1)
	
	#to calculate MEAN_AD
	#MEAN_AD = np.mean(np.abs(buffer_data - np.mean(bufferMatrix)))
	MEANMatrix = np.copy(bufferMatrix)
	firstMean = np.mean(MEANMatrix, axis = 1)
	firstMean = np.reshape(firstMean, (len(firstMean), 1))
	MEANMatrix = np.absolute(MEANMatrix - firstMean)
	MEAN_AD = np.mean(MEANMatrix, axis = 1)
	
	featureMatrix[:,0] = RMS
	featureMatrix[:,1] = PAR
	featureMatrix[:,2] = ZCR
	featureMatrix[:,3] = MAD
	featureMatrix[:,4] = MEAN_AD
	
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
	
	g.write("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s" %(RMS_MEAN, PAR_MEAN, ZCR_MEAN, MAD_MEAN, MEAN_AD_MEAN, RMS_STD, PAR_STD, ZCR_STD, MAD_STD, MEAN_AD_STD, k)) #write to new arff file
	
f.close()