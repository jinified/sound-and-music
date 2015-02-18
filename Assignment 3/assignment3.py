from sys import argv
from scipy.io import wavfile
import numpy as np
import arff

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
num_files = len(fileList)

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
	len = np.size(sampleArray) #number of samples of all files
	RMS = np.sqrt((1/ float(len))*(np.sum(np.square(sampleArray))))
	PAR = (np.max(np.absolute(sampleArray)))/RMS
	
	#to calculate ZCR
	npSign = np.sign(sampleArray)
	npBinary = npSign[1:] * npSign[:-1]
	npNeg = np.array([x for x in npBinary if x < 0])
	npNeg[npNeg < 0] = 1
	ZCR = (1/float(len-1))*(np.sum(npNeg))

	MAD = np.median(np.abs(sampleArray - np.median(sampleArray)))
	f.write("%f,%f,%f,%f,%s" %(RMS, PAR, ZCR, MAD, k)) #write to original arff file

	RMS_data = np.empty_like(sampleArray)
	PAR_data = np.empty_like(sampleArray)
	ZCR_data = np.empty_like(sampleArray)
	MAD_data = np.empty_like(sampleArray)
	MeanAD_data = np.empty_like(sampleArray)
	
	for i in range(0, len/512):
		start = i * 512
		end = (i+2) * 512
		buffer_data = sample[start:end]
		
		RMS = np.sqrt(np.mean(np.square(buffer_data)))
		PAR = (np.max(np.absolute(buffer_data)))/RMS
		
		#to calculate ZCR
		npSign = np.sign(buffer_data)
		npBinary = npSign[1:] * npSign[:-1]
		npNeg = np.array([x for x in npBinary if x < 0])
		npNeg[npNeg < 0] = 1
		ZCR = (1/1023.0)*(np.sum(npNeg))
		
		MAD = np.median(np.abs(buffer_data - np.median(buffer_data)))
		MEAN_AD = np.mean(np.abs(buffer_data - np.mean(buffer_data)))
		
		np.append(RMS_data, RMS)
		np.append(PAR_data, PAR)
		np.append(ZCR_data, ZCR)
		np.append(MAD_data, MAD)
		np.append(MeanAD_data, MEAN_AD)
	
	RMS_MEAN = np.mean(RMS_data)
	PAR_MEAN = np.mean(PAR_data)
	ZCR_MEAN = np.mean(ZCR_data)
	MAD_MEAN = np.mean(MAD_data)
	MEAN_AD_MEAN = np.mean(MeanAD_data)
	
	RMS_STD = np.std(RMS_data)
	PAR_STD = np.std(PAR_data)
	ZCR_STD = np.std(ZCR_data)
	MAD_STD = np.std(MAD_data)
	MEAN_AD_STD = np.std(MeanAD_data)
	
	g.write("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s" %(RMS_MEAN, PAR_MEAN, ZCR_MEAN, MAD_MEAN, MEAN_AD_MEAN, RMS_STD, PAR_STD, ZCR_STD, MAD_STD, MEAN_AD_STD, k)) #write to new arff file
	
f.close()