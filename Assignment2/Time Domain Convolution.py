# Digital-Signal-Processing

import numpy as np
import matplotlib as plt


def myTimeConv(x, h):
    lenx = len(x)
    lenh = len(h)

    fx = np.fft.rfft(lenx)  # fftransform the signals
    fh = np.fft.rfft(lenh)
    # ffx = np.squeeze(np.asarray(fx))
    # ffh = np.squeeze(np.asarray(fh))
    ffx = fx.reshape(1, -1)  # X.shape == (1, 2500)
    ffh = fh.reshape(1, -1)  # Y.shape == (1, 2)
    fxh = np.dot(fx, fh)  # multiply the frequency spectrum of the signals

    y_time = np.fft.irfft(fxh)  # ifftransform back to time-domain
    return (y_time)
   # e = lenx-1
  #  ConvX = np.ones(shape=lenx)
   # for i in range(lenx):
   #     ConvX[i]=x[e]
   #     e=e-1
  #  Total = np.ones(shape=(max(lenx,lenh)) +(2 *(min(lenx,lenh))))
   # j=0

  #  for i in range(len(Total)):
   #     if lenx < i < lenx +lenh:
    #        Total[i] = h[j]
   #         j=j+1
 #   y_time = np.ones(shape=(lenx + lenh-1))
  #  for n in range(lenx + lenh-1):
   #     y_time[n]=np.dot(x, Total[n:n+lenx])



if __name__ == '__main__':



    x = np.ones(shape=200)  # signal
    lengthx= len(x)
    h = np.ones(shape=51)  # impulse response
    lengthh = len(h)
    t1 = np.linspace(0, 1, num=25)
    t2 = np.linspace(1, 0, num=26)
    h = np.append(t1, t2)

    Convolution = myTimeConv(x, h)

    plt.plot(Convolution)
    plt.title("Question 1")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.show()

