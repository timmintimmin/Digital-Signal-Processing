import numpy as np
import scipy.signal as signal
import wave
import soundfile as sf
import struct
from scipy.signal import convolve
from scipy.io import wavfile
from scipy import signal
import scipy
import scipy.signal
import math
import contextlib
from IPython.display import Audio


class LinearWrap(object):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return len(self.it)

    def __setitem__(self, inI, val):
        if type(inI) != int:
            raise RuntimeError('Can only write to integer values')
        self.it[inI] = val

    def __getitem__(self, inI):
        loI = math.floor(inI)
        hiI = math.ceil(inI)
        a = inI - loI
        inRange = lambda val: val >= 0 and val < len(self.it)
        loX = self.it[loI] if inRange(loI) else 0
        hiX = self.it[hiI] if inRange(hiI) else 0
        return loX * (1 - a) + hiX * a


class RingBuffer(object):
    def __init__(self, maxDelay):
        self.maxDelay = maxDelay + 1
        self.buf = np.zeros(self.maxDelay)
        self.writeInd = 0

    def pushSample(self, s):
        self.buf[self.writeInd] = s
        self.writeInd = (self.writeInd + 1) % len(self.buf)

    def delayedSample(self, d):
        d = min(self.maxDelay - 1, max(0, d))
        i = ((self.writeInd + self.maxDelay) - d) % self.maxDelay
        return self.buf[i]


class LinearRingBuffer(RingBuffer):
    def __init__(self, maxDelay):
        self.maxDelay = maxDelay + 1
        self.buf = LinearWrap(np.zeros(self.maxDelay))
        self.writeInd = 0


def damping_filter_coeffs(delays, t_60, alpha):
    element_1 = np.log(10) / 4
    element_2 = 1 - (1 / (alpha ** 2))
    g = np.zeros(len(delays))
    p = np.zeros(len(delays))
    for i in range(len(delays)):
        g[i] = 10 ** ((-3 * delays[i] * (1 / 44100)) / t_60)
        p[i] = element_1 * element_2 * np.log10(g[i])
    print(g)
    print(p)
    return p, g


def delay(input_signal, delay, gain=1):
    output_signal = np.concatenate((np.zeros(delay), input_signal))[:input_signal.size]
    output_signal = output_signal * gain
    return output_signal


def damping_filter(input_signal, p, g):
    B = np.array([g * (1 - p)])
    A = np.array([1, -p])
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(B, A, input_signal)
    return output_signal


def tonal_correction_filter(input_signal, alpha):
    beta = (1 - alpha) / (1 + alpha)
    E_nomin = np.array([1, -beta])
    E_denomin = np.array([1 - beta])
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(E_nomin, E_denomin, input_signal)
    return output_signal


def HighPassFilter():
    sr, x = wavfile.read('Water-Dripping-A4-www.fesliyanstudios.com.wav')  # 16-bit mono 44.1 khz
    b = scipy.signal.firwin(101, cutoff=1000, fs=sr, pass_zero=False)     #the number after cutoff determines cutoff freq
    x = scipy.signal.lfilter(b, [1.0], x)
    wavfile.write('HPFed.wav', sr, x.astype(np.int16))

def LowPassFilter():
    stream = wave.open('Water-Dripping-A4-www.fesliyanstudios.com.wav', "rb")

    n_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    n_frames = stream.getnframes()

    raw_data = stream.readframes(n_frames)  # Returns byte data
    stream.close()

    interpret_wav(raw_data, n_frames, n_channels, sample_width)
    wav_file.close()

def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):



    if sample_width == 1:
        dtype = np.uint8  # unsigned char
    elif sample_width == 2:
        dtype = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


with contextlib.closing(wave.open('Water-Dripping-A4-www.fesliyanstudios.com.wav', 'rb')) as spf:

        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()
        fname = 'Water-Dripping-A4-www.fesliyanstudios.com.wav'
        outname = 'LPFed.wav'

    # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames * nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
        cutOffFrequency = 400.0 # Change this parameter to change cut off frequency!
    # get window size
    # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cutOffFrequency / sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

    # Use moving average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()


def Flanger():
    # Simple Flanger with feedback
    x, sr = sf.read('Water-Dripping-A4-www.fesliyanstudios.com.wav')
    x = LinearWrap(x)

    output = 'Flanged.wav'

    fmod = 0.1
    A = int(0.005 * sr)
    M = int(0.005 * sr)
    BL = 0.7
    FF = 0.7
    FB = -0.7

    if A > M:
        raise RuntimeError("Amplitude of vibrato too high for delay length")

    maxDelaySamps = M + A + 2  # Probably don't need the 2 here, but being safe
    outputSamps = len(x) + maxDelaySamps
    y = np.zeros(outputSamps)
    ringBuf = LinearRingBuffer(maxDelaySamps)
    prevDelaySamp = 0
    deltaPhi = fmod / sr
    phi = 0

    for i in range(outputSamps):
        s = x[i] + prevDelaySamp * FB
        ringBuf.pushSample(s)
        delaySamps = M + int(math.sin(2 * math.pi * phi) * A)
        prevDelaySamp = ringBuf.delayedSample(delaySamps)
        y[i] = s * BL + prevDelaySamp * FF

        phi = phi + deltaPhi
        while phi >= 1:
            phi -= 1

    sf.write(output, y, sr)
    Audio(output)


def Chorus():
    # Simple Chorus
    x, sr = sf.read('Water-Dripping-A4-www.fesliyanstudios.com.wav')
    x = LinearWrap(x)

    output = 'Chorused.wav'

    fmod = 1.5
    A = int(0.002 * sr)
    M = int(0.002 * sr)
    BL = 1.0
    FF = 0.7

    if A > M:
        raise RuntimeError("Amplitude of vibrato too high for delay length")

    maxDelaySamps = M + A + 2  # Probably don't need the 2 here, but being safe
    outputSamps = len(x) + maxDelaySamps
    y = np.zeros(outputSamps)
    ringBuf = LinearRingBuffer(maxDelaySamps)
    deltaPhi = fmod / sr
    phi = 0

    for i in range(outputSamps):
        s = x[i]
        ringBuf.pushSample(s)
        delaySamps = M + int(math.sin(2 * math.pi * phi) * A)
        y[i] = s * BL + ringBuf.delayedSample(delaySamps) * FF

        phi = phi + deltaPhi
        while phi >= 1:
            phi -= 1

    sf.write(output, y, sr)
    Audio(output)


def SpaceAgeReverb():
    #   OPENING WAV FILE   #

    sample_in = 'Water-Dripping-A4-www.fesliyanstudios.com.wav'
    reverb_in = 'THANOS.wav'  # HERE YOU CAN CHANGE HOW BIG YOU WANT TO SOUND TO ALIEN. CHOOSE SMALL.wav, MEDIUM.wav , OR LARGE.wav. OR IF YOU REALLY DARE CHOOSE THANOS.wav
    frame_rate = 44100.0

    wav_file = wave.open(sample_in, 'r')
    num_samples_sample = wav_file.getnframes()
    num_channels_sample = wav_file.getnchannels()
    sample = wav_file.readframes(num_samples_sample)
    total_samples_sample = num_samples_sample * num_channels_sample
    wav_file.close()

    wav_file = wave.open(reverb_in, 'r')
    num_samples_reverb = wav_file.getnframes()
    num_channels_reverb = wav_file.getnchannels()
    reverb = wav_file.readframes(num_samples_reverb)
    total_samples_reverb = num_samples_reverb * num_channels_reverb
    wav_file.close()

    sample = struct.unpack('{n}h'.format(n=total_samples_sample), sample)
    sample = np.array([sample[0::2], sample[1::2]], dtype=np.float64)
    sample[0] /= np.max(np.abs(sample[0]), axis=0)
    sample[1] /= np.max(np.abs(sample[1]), axis=0)

    reverb = struct.unpack('{n}h'.format(n=total_samples_reverb), reverb)
    reverb = np.array([reverb[0::2], reverb[1::2]], dtype=np.float64)
    reverb[0] /= np.max(np.abs(reverb[0]), axis=0)
    reverb[1] /= np.max(np.abs(reverb[1]), axis=0)

    #   MAIN PART OF THE ALGORITHM   #
    # utilize convolution for reverb

    gain_dry = 1
    gain_wet = 1
    output_gain = 0.05

    reverb_out = np.zeros([2, np.shape(sample)[1] + np.shape(reverb)[1] - 1], dtype=np.float64)
    reverb_out[0] = output_gain * (convolve(sample[0] * gain_dry, reverb[0] * gain_wet, method='fft'))
    reverb_out[1] = output_gain * (convolve(sample[1] * gain_dry, reverb[1] * gain_wet, method='fft'))

    #   WRITING TO FILE   #

    reverb_integer = np.zeros((reverb_out.shape))

    reverb_integer[0] = (reverb_out[0] * int(np.iinfo(np.int16).max)).astype(np.int16)
    reverb_integer[1] = (reverb_out[1] * int(np.iinfo(np.int16).max)).astype(np.int16)

    reverb_to_render = np.empty((reverb_integer[0].size + reverb_integer[1].size), dtype=np.int16)
    reverb_to_render[0::2] = reverb_integer[0]
    reverb_to_render[1::2] = reverb_integer[1]

    nframes = total_samples_sample
    comptype = "NONE"
    compname = "not compressed"
    nchannels = 2
    sampwidth = 2

    wav_file_write = wave.open('reverbed.wav', 'w')
    wav_file_write.setparams((nchannels, sampwidth, int(frame_rate), nframes, comptype, compname))

    for s in range(nframes):
        wav_file_write.writeframes(struct.pack('h', reverb_to_render[s]))

    wav_file_write.close()


if __name__ == "__main__":
    SpaceAgeReverb()
    Chorus()
    Flanger()
    LowPassFilter()
    HighPassFilter()
