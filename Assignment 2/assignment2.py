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
	npArray = np.array(sample)
	RMS = np.sqrt((1/ float(num_files))*(np.sum(np.square(npArray))))
	PAR = (np.max(np.absolute(npArray)))/RMS
	npSign = np.sign(npArray)
	npBinary = npSign[1:] * npSign[:-1]
	npNeg = np.array([x for x in npBinary if x < 0])
	npNeg[npNeg < 0] = 1
	ZCR = (1/float(num_files-1))*(np.sum(npNeg))
	MAD = np.median(np.abs(npArray - np.median(npArray)))
	
	f.write("%f,%f,%f,%f,%s" %(RMS, PAR, ZCR, MAD, k))
	
	data = [RMS, PAR, ZCR, MAD]
	if "music" in k:
		features_music[i] = data
	else:
		features_speech[i] = data

plt.plot(features_music[:,2], features_music[:,1])
plt.savefig("music.png")
plt.plot(features_speech[:,2], features_speech[:,1])
plt.savefig("speech.png")

f.close()