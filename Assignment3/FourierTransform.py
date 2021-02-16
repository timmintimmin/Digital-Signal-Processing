
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
    #time = np.arange(0,length_secs, 1)
    samples = (length_secs)*(sampling_rate_Hz)
    CT = 1 / sampling_rate_Hz
    signal = (2 * np.pi * frequency_Hz)
    #signal = np.int16(signal)
    #sine = (amplitude)* np.sin((frequency_Hz)*(time))
    t = np.arange(0,d, CT)

    x = (amplitude)* np.sin((signal * t) + phase_radians)
    t = np.unwrap(t)
    x = np.unwrap(x)

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
    x2= np.sign(np.sin(2*(np.pi)*t*(frequency_Hz)))
    #v(t)= np.sign(np.cos(2*(np.pi)*t*(frequency_Hz)))
    # signal = np.int16(signal)
    # sine = (amplitude)* np.sin((frequency_Hz)*(time))


    x = (amplitude) * np.sin((x2 * t) + phase_radians)
    
    return (t,x)


if __name__ == '__main__':

    t,x = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2)
    print (t)
    print (x)
    plt.plot(t,x)
    plt.title("Question 1")
    plt.ylabel("Generated Signal")
    plt.xlabel("Time")
    plt.show()
    t,x = generateSquare(1.0, 44100, 400, 0.5, np.pi/2)
    print(t)
    print(x)
    plt.plot(t, x)
    plt.title("Question 2")
    plt.ylabel("Generated Signal")
    plt.xlabel("Time")
    plt.show()