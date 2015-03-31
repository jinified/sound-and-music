from scipy.io.wavfile import write
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

fs = 44100.0
sizeOne = 16384
sizeTwo = 2048
freqOne = 100.0
freqTwo = 1234.56
duration = 1.0
amplitude = 1.0
phase = 0.0
freqArray = np.array([freqOne, freqTwo])

def main():
	waveOne = LUTWaves(sizeOne, freqArray)
	waveTwo = LUTWaves(sizeTwo, freqArray)

	wavePerfect = np.array([sin(freqArray[0]), sin(freqArray[1])])
	wavePerfect = np.concatenate((wavePerfect, wavePerfect))

	errorOne = maxError(waveOne, wavePerfect, sizeOne)
	errorTwo = maxError(waveTwo, wavePerfect, sizeTwo)
	errors = np.concatenate((errorOne, errorTwo))
	writeData(errors)
	
def sin(freq):
	period = np.arange(int(duration * fs))
	wave = amplitude * np.sin(period * (freq/fs) * (2 * np.pi) + phase)
	return wave
	
def LUTWaves(size, freqArray):
	samples = (2 * np.pi) * np.arange(size) / size
	LUT = np.sin(samples)
	waves = np.zeros((4, size))	
	for i in range(4):
		if (i < 2):
			delta_phi = freqArray[i] / fs * size
			waves[i] = sansInterpolation(size, delta_phi, LUT)
		else:
			j = i - 2
			delta_phi = freqArray[j] / fs * size
			waves[i] = linearInterpolation(size, delta_phi, LUT)
	return waves

def sansInterpolation(size, delta_phi, LUT):
	phase = 0.0
	buffer = np.zeros(size)
	
	for i in range(size):
		buffer[i] = LUT[int(phase)]
		phase += delta_phi
		if phase >= size:
			phase %= size
	return buffer

def linearInterpolation(size, delta_phi, LUT):
	buffer = np.zeros(size)
	for i in range(size):
		x0 = np.floor(i * delta_phi % size)
		x1 = (x0 + 1) % size
		y0 = LUT[x0]
		y1 = LUT[x1]
		buffer[i] = y0 + (y1 - y0) * ((i * delta_phi % size) - x0) / (x1 - x0)	
	return buffer

def maxError(LUT_sine_wave, perfect_sine_wave, size):
	errors = np.zeros(LUT_sine_wave[:,0].size)
	for i in range(errors.size):
		max_error = np.max(np.abs(LUT_sine_wave[i, :size] - perfect_sine_wave[i, :size]))
		errors[i] = 32767 * max_error
	return errors

def writeData(errors):
	f = open("max_audio_file_error.txt", "w")
	f.write('Frequency\tInterpolation\t16384-sample\t2048-sample\n')
	f.write('100Hz\t\tNo\t\t%s\t%s\n' % (str(errors[0]), str(errors[1])))
	f.write('\t\tLinear\t\t%s\t%s\n' % (str(errors[2]), str(errors[3])))
	f.write('1234.56Hz\tNo\t\t%s\t%s\n' % (str(errors[4]), str(errors[5])))
	f.write('\t\tLinear\t\t%s\t%s\n' % (str(errors[6]), str(errors[7])))

main()