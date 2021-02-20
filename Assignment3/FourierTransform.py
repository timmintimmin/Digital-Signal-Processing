
import numpy as np
import numpy
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def generateSinusoidal(a,b,c,d,e):
    amplitude = a
    sampling_rate_Hz = (b)
    frequency_Hz = (c)
    length_secs = (d)
    phase_radians = (e)
    
 
    #t = np.linspace(0,(length_secs),np.multiply((length_secs)*(sampling_rate_Hz))+1)
    #x = np.sin(2*np.pi*(frequency_Hz)*t)
    
    time = np.arange(0,length_secs, 1)
    samples = (length_secs)*(sampling_rate_Hz)
    CT = 1 / sampling_rate_Hz
    signal = (2 * np.pi * frequency_Hz)
    #signal = np.int16(signal)
    sine = (amplitude)* np.sin((frequency_Hz)*(time))
    t = np.arange(0,d, CT)

    x = (amplitude)* np.sin((signal * t) + phase_radians)
    #t = np.unwrap(t)
    #x = np.unwrap(x)

    return (t,x)

def generateSquare(a,b,c,d,e):
    amplitude = a
    sampling_rate_Hz = (b)
    frequency_Hz = (c)
    length_secs = (d)
    phase_radians = (e)
    # time = np.arange(0,length_secs, 1)
    samples = (length_secs) * (sampling_rate_Hz)
    CT = 1 / sampling_rate_Hz
    signal = (2 * np.pi * frequency_Hz)
    t = np.arange(0, d, CT)
    sinusoidals = (1,3,5,7,9,11,13,15,17,19)
    sumsin = 0
    for number in sinusoidals:
        sumsin = ((1/number)*np.sin(number*2*np.pi*frequency_Hz*t))+ sumsin
    #print ("this is sumsin")
    #print (sumsin)


    x2 = (4/(np.pi)*(sumsin))



    #x2= np.sign(np.sin(2*(np.pi)*t*(frequency_Hz)))
    #x2= np.sign(np.cos(2*(np.pi)*t*(frequency_Hz)))
    
    #signal = np.int16(signal)
    # sine = (amplitude)* np.sin((frequency_Hz)*(time))


    #x = (amplitude) * np.sin((x2 * t) + phase_radians)
    
    return (t,x2)

def computeSpectrum(x, y):
    sample_rate_Hz = y
    N = x.size
    #print ("this is N")
    #print (N)
    #st = 1/(y)   #sampling time
    len_x = len(x)
    spectrum = np.fft.fft((x)/len_x)



    XAbs = np.abs(spectrum)#magnitude spectrum/amplitude
    #print ("this is XAbs")
    #print (XAbs)
    length = len(spectrum)
    middle_index = length//2


    first_half = (spectrum)[:middle_index]
    XPhase = np.angle(spectrum) #phase spectrum
    #print ("this is XPhase")
    #print (XPhase)
    XRe = np.real(first_half) #real part
    #print ("This is XRe")
    #print (XRe)

    XIm = np.imag(first_half)
    #for i in spectrum:
    #    if i in XRe:
    #        continue
    #    else:
    #        XIm.append(i)
        #spectrum[range(int(not (XRe)))] #imaginary
    #print("This is XIm")
    #print(XIm)
    f = np.arange(0, N)*y/N
    #frequency of bins
    #print ("this is f")
    #print (f)

    return (XAbs, XPhase, XRe, XIm, f)


def generateBlocks(a, b, c, d):
    x = (a)
    sample_rate_Hz = (b)
    block_size = (c)
    hop_size = (d)
    lengthx = len(x)
    print (lengthx)
    #createblocks = [x[i] for i in range(0, len(x), hop_size)]
    createblocks = [x[i:i +block_size] for i in range(0, len(x), hop_size)]
    if len(x) < (block_size):
            createblocks = np.pad(i,(0,(len(x)-t))) #zero padding
    print ('this is cb')
    print (createblocks)
    #sig_splits = []
    #for i in xrange(0, len(x), int((seconds - overlap) * rate)):
    #    split = sig[i:i + int(seconds * rate)]

    #    # End of signal?
    #    if len(split) < int(minlen * rate):
    #        break

        # Signal chunk too short?
    #    if len(split) < int(rate * seconds):
    #        split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))

    #    sig_splits.append(split)
    t = [item[0] for item in (createblocks)]
    print ('this is t')
    print (t)
    lengtht= len(t)
    print ('this is length t')
    print (lengtht)
    each = np.array(1)
    X_array =[]
    for i in (createblocks):
        x = i
        X_array.append(x)
    #m1 = createblocks[1]
    #r1 = np.array([2048,[m1]], dtype=object)
    #m2 = createblocks[2]
    #r2 = np.array([2048,[m2]], dtype=object)
   # x = np.concatenate(r1,r2)
        #for item[2] in x:
       # for i in createblocks:
        #    item[2]= (i)
    #for i in createblocks:
     #   x[2048,i] = np.hstack(i)
        #each = (i)
        #x = np.array([(block_size), (each)])
    x = (X_array)

    print ('this is x')

    print (x)
    print ('this is lengthx')

    return (t,x)


    #t = #time stamps of blocks
    # x = #matrix (blocksize x N) where each column is a block of the signal)


def plotSpecgram(freq_vector, time_vector, magnitude_spectrogram):
    if len(freq_vector) < 2 or len(time_vector) < 2:
        return

    Z = 20. * np.log10(magnitude_spectrogram)
    Z = np.flipud(Z)

    pad_xextent = (time_vector[1] - time_vector[0]) / 2
    xmin = np.min(time_vector) - pad_xextent
    xmax = np.max(time_vector) + pad_xextent
    extent = xmin, xmax, freq_vector[0], freq_vector[-1]

    im = plt.imshow(Z, None, extent=extent,
                    origin='upper')
    plt.axis('auto')
    plt.show()


def mySpecgram(a,b,c,d,e):
    global freq_vector
    x = (a)
    block_size = (b)
    hop_size = (c)
    sampling_rate_Hz = (d)
    window_type = (e)
    each = np.array(1)
    lengthx = len(x)
    #rectangle =
    winSize = (block_size)
    winCosine = np.zeros(winSize)
    winHamming = np.zeros(winSize)

    for i in range(0,winSize):
        t = float(i)/ float(winSize)
        t = t - 0.5
        winCosine[i] = math.cos(math.pi * t)
        winHamming[i] = (25.0/46.0) + (21.0/46.0) * math.cos(math.pi *2 *t)
    q,w = generateBlocks(x,sampling_rate_Hz, block_size, hop_size)
    print ('this is q')
    print (q)
    N = len(q)
    window = np.hanning(N)
    print (window)
    print ('this is window')
    if "rect" in e:
        window = winCosine
    else:
        window = winHamming
    windowed = []
    for i in (w):
        w = i
        windowed = np.multiply(window, w)
    magnitude_spectrogram = np.array([(block_size / 2)])
    freq_vector = np.array([(block_size) / 2, 1])
    time_vector = np.array([block_size / 2, N])
    time_vector = np.append(time_vector, q)
    a,b,c,d,e = computeSpectrum(x,x)
    for i in windowed:
        each = (i)
        a,b,c,d,e = computeSpectrum(i, sampling_rate_Hz)
        freq_vector = np.append(freq_vector, e)
        magnitude_spectrogram= np.append(magnitude_spectrogram, a)


    print ("this is mag")
    print (magnitude_spectrogram)



    #plt.figure()

    mag = (magnitude_spectrogram)
    freq = (e)
    #plot = plotSpecgram(freq_vector, time_vector, mag)
    plt.plot(time_vector, mag)
    plt.title("Frequency response of Hanning window")
    plt.ylabel("Magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.axis("tight")

    return (freq_vector, time_vector, magnitude_spectrogram)


if __name__ == '__main__':

    t,x = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2)
    print (t)
    print (x)
    
    plt.plot(t,(x))
    plt.xlim(0,0.005)
    plt.title("Question 1")
    plt.ylabel("Generated Signal")
    plt.xlabel("Time")
    plt.show()
    t,x2 = generateSquare(1.0, 44100, 400, 0.5, 0)
    print(t)
    print ("this is x2")
    print(x2)
    plt.plot(t, x2)
    plt.xlim(0,0.005)
    plt.title("Question 2")
    plt.ylabel("Generated Signal")
    plt.xlabel("Time")
    plt.show()
    print ("this is X")
    print (x)
    lennx = len(x)
    print (lennx)
    a, b, c, d, e = computeSpectrum((x),44100)
    a2, b2, c2, d2, e2 = computeSpectrum((x2),44100)
    lengthx= len(e)
    print (lengthx)
    #a = padarray([a], 25000)
    #e = padarray([e], 25000)
    #a = np.pad((a),(0,22050, 'constant'))
    print ("This is a")
    print (a)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Sinusoidal - Magnitude')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude Spectra')
    #ax1.set_xlim(43600,43800)
    ax1.plot(e, a)
    ax2.set_title('Sinusoidal - Phase')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Phase Spectra')
    ax2.set_xlim(0,4000)
    ax2.plot(e, b)
    fig, (ax3, ax4) = plt.subplots(1, 2)
    ax3.set_title('Square - Magnitude')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Magnitude Spectra')
    #ax3.set_xlim(0,1400)
    ax3.plot(e2, a2)
    ax4.set_title('Square - Phase')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Phase Spectra')
    ax4.set_xlim(0,4000)
    ax4.plot(e2, b2)
    plt.show()
    t,x = generateBlocks(x, 44100, 2048, 1024)
    a3,b3,c3 = mySpecgram(x2, 2048, 1024, 44100, 'hann')



#Sine Sweep
x = SineSweep(t, f0, t1, f1)
# t = timebase array
#f0 = frequency at time 0
#f1 = freqeuncy at time 1

t0 = t(1)
T = t1 - t0
k = (f1 - f0) / T
x = cos(2 * pi * (k / 2 * t + f0). * t)