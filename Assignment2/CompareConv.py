# Digital-Signal-Processing

import numpy as np
import scipy as sp
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
import scipy.signal
import statistics
#The length of Y, when length of x = 200, and the length of h =100, would equal 400.
def myTimeConv(x,h):



    lenx = len(x)
    lenh = len(h)

    #e = lenx-1
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

    #ConvX = np.reshape(ConvX, 169601)
    ConvX =np.array(np.array(ConvX))
    ConvX = np.pad(ConvX, 2)
    Total = np.array(np.array(Total))
    ConvX = np.pad(Total, 2)
    y_time = np.zeros(shape=(lenx + lenh-1))
    y_time = np.array(np.array(y_time))

    #np.flip(ConvX, 1)
    #t = np.linspace(max(lenx,lenh) + (min(lenx,lenh)),Total)
    for n in range(lenx+lenh-1):
        y_time[n]=np.multiply(ConvX[n],Total[n])


    return y_time





def readfile(filename):
    samplerate, data = wavfile.read(filename)
    return data

def ScipyCon(x,h):


    x1 = np.array(np.array(x))  # signal
    h1 = np.array(np.array(h))  # impulse response
    w = sp.signal.convolve(x1,h1)
    return w


#the average of the absolute difference
def meanabsolutedifference(x,h):
    m = np.zeros(shape=(len(x)))
    for i in range(len(x)):
        m[i]=abs(x[i]-h[i])
        masd =np.sum(m)/(len(x)*2)
        return masd


def meandifference(x,h):
    SX = np.sum(x)
    mx = SX / len(x)
    sh = np.sum(h)
    mh = sh / len(h)
    mad = mx - mh
    return mad

def CompareConv(a,b):

    md = meandifference(a,b)
    print (md)
    mad = meanabsolutedifference(a,b)
    print (mad)

    c = a-b
    sc = statistics.stdev(c)
    #ss = statistics.stdev(b)
   # stdev = sc - ss
    print (c)

    Timedifference = f-v
    print (Timedifference)



if __name__ == '__main__':

    x = readfile('./audio/piano.wav')
    h = readfile('./audio/impulse-response.wav')
    #x = np.zeros(shape=200)  # signal
    lengthx= len(x)
    #h = np.zeros(shape=51)  # impulse response
    lengthh = len(h)
    #t1 = np.linspace(0, 1, num=25)
    #t2 = np.linspace(1, 0, num=26)
    #h = np.append(t1, t2)
    t = time.time()
    Convolution = myTimeConv(x, h)
    q = time.time()
    f = q-t
    o = time.time()
    SC = ScipyCon(x, h)
    k = time.time()
    v = k - o
    Finale = CompareConv(Convolution,SC)




