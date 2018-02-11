import matplotlib.pyplot as plot

from scipy.io import wavfile

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('test3.wav')

samplingFrequency1, signalData1 = wavfile.read('test2.wav')

samplingFrequency2, signalData2 = wavfile.read('test1.wav')

# Plot the signal read from wav file

plot.subplot(311)

plot.title('Spectrogram of a wav file with piano music')

plot.specgram(signalData1, Fs = samplingFrequency1, mode = 'default')

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(312)

plot.specgram(signalData, Fs = samplingFrequency, mode = 'default')

plot.xlabel('Time')

plot.ylabel('Frequency')

plot.subplot(313)

plot.specgram(signalData2, Fs = samplingFrequency2, mode = 'default')

plot.xlabel('Time')

plot.ylabel('Frequency')

plot.show()