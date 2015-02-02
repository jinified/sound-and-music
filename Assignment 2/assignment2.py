from sys import argv
from scipy.io import wavfile
import numpy as np
import arff
import matplotlib.pyplot as plt
#import pylab

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
num_files = len(fileList)

f = open("assignment2.arff", "w")
f.write('''@RELATION music_speech
@ATTRIBUTE RMS NUMERIC
@ATTRIBUTE PAR NUMERIC
@ATTRIBUTE ZCR NUMERIC
@ATTRIBUTE MAD NUMERIC
@ATTRIBUTE class {music,speech}\n
@DATA\n''')

features_music = np.zeros((num_files,4))
features_speech = np.zeros((num_files,4))

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
	
	data = [RMS, PAR, ZCR, MAD]
	if "music" in k:
		features_music[i] = data
	else:
		features_speech[i] = data

ZCRplot = plt.scatter(features_music[:,2], features_music[:,1], c = "red", marker = "o")
PARplot = plt.scatter(features_speech[:,2], features_speech[:,1], c = "blue", marker = "x")
plt.xlabel("ZCR")
plt.ylabel("PAR")
plt.legend((ZCRplot, PARplot), ('ZCR', 'PAR'), loc = 4)
plt.savefig("ZCRxPAR.png")

plt.clf() #clear the figure

RMSplot = plt.scatter(features_music[:,0], features_music[:,3], c = "green", marker = "^")
MADplot = plt.scatter(features_speech[:,0], features_speech[:,3], c = "orange", marker = "s")
plt.xlabel("RMS")
plt.ylabel("MAD")
plt.legend((RMSplot, MADplot), ('RMS', 'MAD'), loc = 4)
plt.savefig("RMSxMAD.png")

f.close()