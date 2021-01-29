# Digital-Signal-Processing
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def readfile(filename):

    samplerate, data = wavfile.read(filename)
    return data[:,0]
def crossCorr(x,y):
    z = np.correlate(x,y)
    return z


if __name__ == '__main__':

    x = readfile('drum_loop.wav')
    lengthx = len(x)
    y = readfile('snare.wav')
    lengthy = len(y)
    z = crossCorr(x,y)
    plt.plot(z)
    plt.show()