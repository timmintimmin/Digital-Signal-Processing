
# Digital-Signal-Processing

import numpy as np
import matplotlib

def myTimeConv(x,h):
# make simple rectangular signals
    x1 = np.array(x)    #signal
    h1 = np.array(h)    #impulse response
    #r1[1:20] = 1.
    #r2[20:40] = 1.

    fft_x = np.fft.rfft(x1)           # fftransform the signals
    fft_h = np.fft.rfft(h1)

    fft_xh = np.multiply(fft_x, fft_h)  # multiply the frequency spectrum of the signals

    y = np.fft.irfft(fft_xh)            # ifftransform back to time-domain
    return y
    y.shape

if __name__ == '__main__':

    conv = myTimeConv(200,100)
    conv.shape


