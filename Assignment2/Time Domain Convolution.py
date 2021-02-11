
# Digital-Signal-Processing

import numpy as np
import matplotlib

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

    y1 = np.fft.irfft(fxh) # ifftransform back to time-domain
    y = np.ones(y1)
    return y


if __name__ == '__main__':

    x = (200,1)
    h = (100,1)
    # x = readfile('drum_loop.wav')
    # lengthx = len(x)
    # y = readfile('snare.wav')
    # lengthy = len(y)
    Convolution = myTimeConv(x,h)
    print (Convolution)

