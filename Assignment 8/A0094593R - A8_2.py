import numpy
import math
import scipy.io.wavfile
import scipy.signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

submission_file = "Max Audio File Errors.txt"

SAMPLING_FREQ = 44100.0

SET_ONE_NUM_SAMPLES = 16384
SET_TWO_NUM_SAMPLES = 2048

WAVE_DURATION = 1.0

freq_to_test = numpy.array([100.0, 1234.56])
    
def create_sine_wave(freq, duration = WAVE_DURATION, amplitude = 1.0, phase = 0.0):
    t = numpy.arange(int(duration * SAMPLING_FREQ))
    return amplitude * numpy.sin(t * (freq / SAMPLING_FREQ) * (2 * numpy.pi) + phase)
    
def draw_wave(wave, x_limit = 0):
    image = plt.figure(figsize=(17.0, 7.0))
    axes = image.add_subplot(111)
    
    plt.xlim(xmax = x_limit)
        
    axes.plot(wave)
    image.tight_layout()

def get_lut_waves(num_samples, freq_to_test):
    lut = create_lookup_table(num_samples)
    
    wave_sets = numpy.zeros((freq_to_test.size * 2, num_samples))
    
    for i in range(freq_to_test.size):
        phase_increment = freq_to_test[i] / SAMPLING_FREQ * num_samples
        wave_sets[i] = create_new_buffer_no(num_samples, phase_increment, lut)
    
    for i in range(freq_to_test.size):
        phase_increment = freq_to_test[i] / SAMPLING_FREQ * num_samples
        wave_sets[i+2] = create_new_buffer_linear(num_samples, phase_increment, lut)
    
    return wave_sets

def create_lookup_table(num_samples):
    sample_array = (2 * numpy.pi) * numpy.arange(num_samples) / num_samples
    lookup_table = numpy.sin(sample_array)
    
    return lookup_table

def create_new_buffer_no(num_samples, phase_increment, lut):
    buffer = numpy.zeros(num_samples)
    
    for i in range(num_samples):
        buffer[i] = lut[round(i * phase_increment) % num_samples]
            
    return buffer

def create_new_buffer_linear(num_samples, phase_increment, lut):
    buffer = numpy.zeros(num_samples)
    
    for i in range(num_samples):
        x_0 = math.floor(i * phase_increment % num_samples)
        x_1 = (x_0 + 1) % num_samples
    
        y_0 = lut[x_0]
        y_1 = lut[x_1]
        buffer[i] = y_0 + (y_1 - y_0) * ((i * phase_increment % num_samples) - x_0) / (x_1 - x_0)
    
    return buffer

def get_max_audio_file_error(lut_sin_wave, perf_sin_wave, num_samples):
    errors = numpy.zeros(lut_sin_wave[:,0].size)
    
    for i in range(errors.size):
        max_error = numpy.max(numpy.abs(lut_sin_wave[i, :num_samples] - perf_sin_wave[i, :num_samples]))
        errors[i] = 32767 * max_error
        
    return errors

def write_file(file_writer, all_errors):
    file_writer.write('Frequency\tInterpolation\t16384-sample\t2048-sample\n')
    file_writer.write('100Hz\t\tNo\t\t\t\t' + str(all_errors[0]) + '\t' + str(all_errors[1]) + '\n')
    file_writer.write('\t\t\tLinear\t\t\t' + str(all_errors[2]) + '\t' + str(all_errors[3]) + '\n')
    file_writer.write('1234.56Hz\tNo\t\t\t\t' + str(all_errors[4]) + '\t' + str(all_errors[5]) + '\n')
    file_writer.write('\t\t\tLinear\t\t\t' + str(all_errors[6]) + '\t' + str(all_errors[7]) + '\n')

set_one_lut_sin_wave = get_lut_waves(SET_ONE_NUM_SAMPLES, freq_to_test)
set_two_lut_sin_wave = get_lut_waves(SET_TWO_NUM_SAMPLES, freq_to_test)

perf_sin_wave = numpy.array([create_sine_wave(freq_to_test[0]), create_sine_wave(freq_to_test[1])])
perf_sin_wave = numpy.concatenate((perf_sin_wave, perf_sin_wave))

set_one_errors = get_max_audio_file_error(set_one_lut_sin_wave, perf_sin_wave, SET_ONE_NUM_SAMPLES)
set_two_errors = get_max_audio_file_error(set_two_lut_sin_wave, perf_sin_wave, SET_TWO_NUM_SAMPLES)

all_errors = numpy.concatenate((set_one_errors, set_two_errors))

file_writer = open(submission_file, 'w')
write_file(file_writer, all_errors)
file_writer.close()