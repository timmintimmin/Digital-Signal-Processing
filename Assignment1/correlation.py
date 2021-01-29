# Digital-Signal-Processing
import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
    samplerate, data = wavfile.read(filename)
    return data[:,0]
def crossCorr(x,y):
    lengthx = len(x)
    lengthy = len(y)


if __name__ == '__main__':

    x=readfile('./Assignment01/drum_loop.wav')
    lengthx = len(x)
    y = readfile('./Assignment01/snare.wav')
    lengthy = len(y)
    z = crossCorr(x,y)
    plt.plot(z)
    plt.show()