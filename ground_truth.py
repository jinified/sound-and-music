from sys import argv
from scipy.io import wavfile
import numpy

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
result = []
for i in fileList:
	j, k = i.split("\t") #split string after \t
	rate, sample = wavfile.read(j) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	npArray = numpy.array(sample)
	len = numpy.size(npArray)
	RMS = numpy.sqrt((1/ float(len))*(numpy.sum(numpy.square(npArray))))
	result.append(RMS)
	result.append((numpy.argmax(numpy.abs(npArray)))/RMS) #calculate PAR
	result.append((1/(len-1))(#something))#calculate ZCR
	result.append(numpy.median(numpy.abs(npArray - numpy.median(npArray))))#calculate MAD
	