# Digital-Signal-Processing
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


def similarity(template, test):
    corr = fftconvolve(template, test, mode='same')

    return max(abs(corr))

def readfile(filename):

    samplerate, data = wavfile.read(filename)
    return data[:,0]
def crossCorr(x,y):
    z = np.correlate(x,y)
    return z

def findSnarePosition(filename1,filename2):
    r = crossCorr(x,y)
    maxr=max(r)
    print(maxr)
    for i in range(len(r)):
        if r[i]>=maxr-0.1:
            n = 4
            print (r[np.argsort(r)[-n:]])



    #maxcor = np.max(np.abs(filename1,filename2))
   # return maxcor
    #c1 = corr.abs().unstack()
    #c1.sort_values(ascending=False)
    #return maxcor

if __name__ == '__main__':

    x = readfile('drum_loop.wav')
    lengthx = len(x)
    y = readfile('snare.wav')
    lengthy = len(y)
    z = crossCorr(x,y)

    maxcor = findSnarePosition('drum_loop.wav','snare.wav')

