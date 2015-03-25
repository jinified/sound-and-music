from scipy.io import wavfile
import scipy
import scipy.fftpack
import math
import numpy as np
import matplotlib.pyplot as plt

midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]
samplingRate = 8000.0
duration = 0.25
amplitude = 0.25
phase = 0.0
samples = samplingRate * duration

def main():
	music = np.zeros(0)
	musicADSR = np.zeros(0)

	ADSR = ADSR_envelope()    
	for i in midis:
		freq = fundamental_freq(i)
		node = np.zeros((4, samples))
		for j in range (1, 5):
			node[j-1] = sin(freq * j)
		sumNodes = np.sum(node, axis = 0)
		music = np.concatenate((music, sumNodes))
		musicADSR = np.concatenate((musicADSR, sumNodes * ADSR))
    
	(32767 * music).astype(np.int16)
	(32767 * musicADSR).astype(np.int16)
	scipy.io.wavfile.write("notes.wav", int(samplingRate), music)
	scipy.io.wavfile.write("notes-adsr.wav", int(samplingRate), musicADSR)
	plot(music, "spectrogram-notes.png")
	plot(musicADSR, "spectrogram-notes-adsr.png")

def ADSR_envelope():
	attack = 0.1 * samples
	decay = 0.15 * samples
	sustain = 0.375 * samples 
	release = 0.375 * samples
	
	A = np.linspace(0.0, 1.0, attack)
	D = np.linspace(1.0, 0.5, decay)
	S = np.linspace(0.5, 0.5, sustain)
	R = np.linspace(0.5, 0.0, release)
	
	envelope = np.concatenate((A, D, S, R))
	return envelope
	
def fundamental_freq(note):
	if (note == 0):
		return 0.0
	else:
		freq = 440 * (pow(2, ((note - 69) / 12.0)))
		return freq

def sin(freq, duration = duration, amplitude = amplitude, phase = phase):
	period = np.arange(int(duration * samplingRate))
	wave = amplitude * np.sin(period * (freq/samplingRate) * (2 * np.pi) + phase)
	return wave

def magSpectrum (notes, window):
	fft = scipy.fftpack.fft(notes * window)
	fft = fft[:len(fft) / 2 + 1]
	magfft = np.abs(fft) / np.sum(window) / 2.0
	epsilon = pow(10, -10)
	mag = 20 * np.log10(magfft + epsilon)
	return mag

def plot(notes, fileName):
	numBuffers = int(len(notes) / 256 - 1)
	buffers = np.zeros((numBuffers, 257))
	
	for i in range(numBuffers):
		start = i * 256
		end = i * 256 + 512
		slice = notes[start:end]
		buffers[i] = magSpectrum(slice, np.blackman(512))
		
	buffers = buffers.transpose()
	plt.title('MIDI Spectogram')
	plt.xlabel('Time(hops)')
	plt.ylabel('Frequency bin')
	plt.imshow(buffers, origin='lower')
	plt.savefig(fileName)
	
main()