# Digital-Signal-Processing

import numpy as np
import scipy as sp
from scipy.io import wavfile
import matplotlib.pyplot as plt

#The length of Y, when length of x = 200, and the length of h =100, would equal 400.
def myTimeConv(x,h):
    lenx = len(x)
    lenh = len(h)

    print (lenx)
    print (lenh)
    e = lenx-1
    ConvX = np.zeros(shape=lenx)
    for i in range(lenx):
         ConvX[i]=1
         #ConvX[i]=x[e]
         #e=e-1
    Total = np.zeros(shape=(max(lenx,lenh)) +(2 *(min(lenx,lenh))))
    j=0
    for i in range(len(Total)):
        if lenx < i < lenx +lenh:
            Total[i] = h[j]
            j=j+1
    ConvX = np.pad(ConvX, 51)
    print (ConvX.shape)

    print (ConvX)
    print (Total)
    ConvX = np.reshape(ConvX, 302)
    ConvX =np.array(np.array(ConvX))
    Total = np.array(np.array(Total))
    y_time = np.zeros(shape=(lenx + lenh-1))
    #np.flip(ConvX, 1)
    for n in range(lenx + lenh -1):
        y_time[n]=np.multiply(ConvX[n],Total[n])
    return y_time


if __name__ == '__main__':

    x = np.zeros(shape=200)  # signal
    lengthx= len(x)
    h = np.zeros(shape=51)  # impulse response
    lengthh = len(h)
    #j=0.04
    #for i in range (lengthx):
    #    x[i]=1
    #for i in range (lengthh):
    #    if i<=25:
    #        h[i]=i*j
    #    else:
    #        h[i]=1-(i-25)*j

    t1 = np.linspace(0, 1, num=25)
    t2 = np.linspace(1, 0, num=26)
    h = np.append(t1, t2)
    print (h)

    Convolution = myTimeConv(x, h)

    plt.plot(Convolution)
    plt.title("Question 1")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.show()


def readfile(filename):

    samplerate, data = wavfile.read(filename)
    return data[:,0]





def ScipyCon(x,h)
    x1 = np.ones(x)  # signal
    h1 = np.ones(h)  # impulse response
    w = sp.signal.convolve(x1.h1)
    return w

def ConmpareConv(x,h):




#the average will need this in order to calculate the other mean stuff
def mean(x):
        SX = np.sum(x)
        mx = sx/lex(x)
        return mx

#the average of the absolute difference
def meanabsolutedifference(x,h):
    m = np.zeros(shape=(len(x)))
    for i in range(len(x)):
        m[i]=abs(x[i]-h[i])
        masd =np.sum(d)/(len(x)*2)
        return masd


def meandifference(x,h):
    mx = mean(x)
    mh = mean(h)
    mad = mx - mh
    return mad


if __name__ == '__main__':

    x = loadSoundfile(impulse-response.wav)
    y = loadSoundfile(piano.wav)
    lenx = len(x)
    leny = len(y)

    t1 = np.linspace(0,1,num=25)
    t2 = np.linspace(1,0,num=26)
    h = np.append(t1,t2)
    print (h)

    x = (200,1)
    Convolution = myTimeConv(x,h)
    print (Convolution)

