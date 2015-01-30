from sys import argv
from scipy.io import wavfile
import numpy
import csv

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
b = open('assignment1.csv', 'wb')
for i in fileList:
	j, k = i.split("\t") #split string after \t
	rate, sample = wavfile.read(j) #read in wavfile
	sample = sample / 32768.0 #convert sample to floats
	npArray = numpy.array(sample)
	len = numpy.size(npArray)
	RMS = numpy.sqrt((1/ float(len))*(numpy.sum(numpy.square(npArray))))
	PAR = (numpy.max(numpy.absolute(npArray)))/RMS
	npSign = numpy.sign(npArray)
	npBinary = npSign[1:] * npSign[:-1]
	npNeg = numpy.array([x for x in npBinary if x < 0])
	npNeg[npNeg < 0] = 1
	ZCR = (1/float(len-1))*(numpy.sum(npNeg))
	MAD = numpy.median(numpy.abs(npArray - numpy.median(npArray)))
	
	a = csv.writer(b)
	data = [j, RMS, PAR, ZCR, MAD]
	a.writerow(data)
b.close()