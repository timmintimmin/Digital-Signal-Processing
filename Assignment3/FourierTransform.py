
import numpy as np
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy



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
    print ("this is sumsin")
    print (sumsin)


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
    print ("this is N")
    print (N)
    st = 1/(y)   #sampling time
    len_x = len(x)
    spectrum = np.fft.fft((x)/len_x)



    XAbs = np.abs(spectrum)#magnitude spectrum/amplitude
    print ("this is XAbs")
    print (XAbs)
    XPhase = np.angle(spectrum) #phase spectrum
    print ("this is XPhase")
    print (XPhase)
    XRe = spectrum[range(int((len_x)/2))] #real part
    print ("This is XRe")
    print (XRe)
    XIm = []
    for i in spectrum:
        if i in XRe:
            continue
        else:
            XIm.append(i)
    return XIm
        #spectrum[range(int(not (XRe)))] #imaginary
    print("This is XIm")
    print(XIm)
    f = np.arange(0, N)*y/N
    #frequency of bins
    print ("this is f")
    print (f)

    return (XAbs, XPhase, XRe, XIm, f)



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
    (a, b, c, d, e) = computeSpectrum((x),44100)
    #(XAbs, XPhase, XRe, XIm, f) = computeSpectrum((x2),44100)
