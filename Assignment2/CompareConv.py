# Digital-Signal-Processing

import numpy as np
import scipy as sc
import matplotlib




def ScipyCon(x,h)
    x1 = np.ones(x)  # signal
    h1 = np.ones(h)  # impulse response
    sc.signal.convolve(x,h)
def ConmpareConv(x,h):


def myTimeConv(x,h):
# make simple rectangular signals

    x1 = np.ones(x)  #signal
    h1 = np.ones(h)    #impulse response
    print (x1)
    print (h1)
    fx = np.fft.rfft(x1)           # fftransform the signals
    fh = np.fft.rfft(h1)
   # ffx = np.squeeze(np.asarray(fx))
    #ffh = np.squeeze(np.asarray(fh))
   # ffx = fx.reshape(1, -1) # X.shape == (1, 2500)
  #  ffh = fh.reshape(1, -1) # Y.shape == (1, 2)
    fxh = np.dot(fx,fh)  # multiply the frequency spectrum of the signals

    y_time = np.fft.irfft(fxh) # ifftransform back to time-domain

    return y_time



if __name__ == '__main__':



    t1 = np.linspace(0,1,num=25)
    t2 = np.linspace(1,0,num=26)
    h = np.append(t1,t2)
    print (h)

    x = (200,1)
    Convolution = myTimeConv(x,h)
    print (Convolution)

