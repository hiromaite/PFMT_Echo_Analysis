import numpy as np
from scipy import signal
from scipy.stats import norm


def gaussian_filter(data, samplerate, freq):
    # get the convolution integral with the Gaussian function
    x_gauss = np.arange(-10, 10, 6 * freq / samplerate)
    weight = norm.pdf(x_gauss)
    out = np.convolve(data, weight, mode='same') / np.sum(weight)
    return out


def highpass_filter(data, samplerate, freq_pass, freq_stop):
    freq_n = samplerate / 2  # nyquist frequency
    wp = freq_pass / freq_n
    ws = freq_stop / freq_n
    gain_pass = 3
    gain_stop = 40
    n, Wn = signal.buttord(wp, ws, gain_pass, gain_stop)
    b, a = signal.butter(n, Wn, "high")
    out = signal.filtfilt(b, a, data)
    return out


def lowpass_filter(data, samplerate, freq_pass, freq_stop):
    freq_n = samplerate / 2  # nyquist frequency
    wp = freq_pass / freq_n
    ws = freq_stop / freq_n
    gain_pass = 3
    gain_stop = 40
    n, Wn = signal.buttord(wp, ws, gain_pass, gain_stop)
    b, a = signal.butter(n, Wn, "low")
    out = signal.filtfilt(b, a, data)
    return out

 
def attenuation_coefficient(data, attn_order):
    # get attenuation coefficient from integrated values
    size_data = len(data)
    int_data = np.zeros(size_data, float)
    int_data[0] = data[0]
    for i in range(size_data - 1):
        int_data[i + 1] = int_data[i] + data[i + 1]
    int_data = int_data / int_data[-1]
    out = int_data ** attn_order
    return out


