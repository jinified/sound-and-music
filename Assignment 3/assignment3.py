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
	f.write("%f,%f,%f,%f,%s" %(RMS, PAR, ZCR, MAD, k)) #write to arff file

	for i in range(len)
		start = i * 512
		end = (i+2) * 512
		buffer_data = sample[start:end]
		
		RMS = np.sqrt((1/float(1024))*(np.sum(np.square(buffer_data))))
		PAR = (np.max(np.absolute(buffer_data)))/RMS
	
		#to calculate ZCR
		npSign = np.sign(buffer_data)
		npBinary = npSign[1:] * npSign[:-1]
		npNeg = np.array([x for x in npBinary if x < 0])
		npNeg[npNeg < 0] = 1
		ZCR = (1/float(1024-1))*(np.sum(npNeg))
		
		MAD = np.median(np.abs(buffer_data - np.median(buffer_data)))
		MEAN_AD = np.mean(np.abs(buffer_data - np.mean(buffer_data)))
		
f.close()