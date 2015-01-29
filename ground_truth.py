from sys import argv
from scipy.io import wavfile

script, filename = argv

txt = open(filename)
fileList = txt.readlines()
for i in fileList:
	j, k = i.split("\t")
	rate, sample = wavfile.read(j)
	print sample
	