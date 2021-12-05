import tkinter
from tkinter import filedialog, messagebox
import os

import cv2
import numpy as np
import scipy.io
from scipy import signal
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt


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


flg_test_graph_out = False
# osciloscope timescale setting is 1dev = 20us
sample_rate = 1250 / (20 * 0.001 * 0.001)

# Open file select dialog, check file exist
root = tkinter.Tk()
root.withdraw()
fTyp = [('MATLAB file', '*.mat')]
iDir = os.path.abspath(os.path.dirname(__file__))
files = filedialog.askopenfilenames(filetypes=fTyp, initialdir=iDir)
try:
    stat = os.stat(files[0])  # do try-catch with first file
except:
    messagebox.showinfo('Information',
                        "Execution has canceled in dialog window")
    exit()

for file in tqdm(files, leave=False):
    path = os.path.dirname(file)
    filename = os.path.splitext(os.path.basename(file))[0]
    mat = scipy.io.loadmat(file)
    multi_data = mat['multiData']

    # acquire data shape
    num_rec, rec_len = multi_data.shape

    # time scale
    x = np.linspace(0, rec_len - 1, rec_len)
    x = x - 0.1 * rec_len  # trigger position
    x = x / sample_rate
    x = x * 1520 * 1000  # speed of sound

    out_data = np.zeros((num_rec, rec_len), float)

    for rec in tqdm(range(num_rec), leave=False):
        # get the convolution integral with the Gaussian function
        freq = 2 * 1000 * 1000
        conv_data = gaussian_filter(multi_data[rec], sample_rate, freq)

        # high-pass filter
        fp = 2 * 1000 * 1000
        fs = 1 * 1000 * 1000
        hpf_data = highpass_filter(conv_data, sample_rate, fp, fs)

        # absolute
        abs_hpf_data = abs(hpf_data)

        # get the convolution integral with the Gaussian function
        freq = 0.2 * 1000 * 1000
        conv_abs_hpf_data = gaussian_filter(abs_hpf_data, sample_rate, freq)

        # get envelope wave
        env_hpf_data = abs(signal.hilbert(hpf_data))

        # get the convolution integral with the Gaussian function
        freq = 0.2 * 1000 * 1000
        conv_env_hpf_data = gaussian_filter(env_hpf_data, sample_rate, freq)

        """ ratio = conv_data_3[200] / conv_data_2[200]
        conv_data_2 = conv_data_2 * ratio """

        # normalized by min & max value while region of Rx signals from body
        region_start = int(0.3 * rec_len)
        region_end = -1 * int(0.1 * rec_len)
        min_data = min(conv_env_hpf_data[region_start: region_end])
        max_data = max(conv_env_hpf_data[region_start: region_end])
        norm_conv_env_hpf_data = (
            conv_env_hpf_data - min_data) / (max_data - min_data)

        # get integrated values
        int_data = norm_conv_env_hpf_data.copy()
        for i in range(rec_len - 1):
            int_data[i + 1] = int_data[i] + norm_conv_env_hpf_data[i + 1]
        int_data = int_data / int_data[-1]

        # make attenuation function
        db = (x / 10) * -1
        db = np.multiply(db, int_data)
        attn = 10 ** (db / 20)
        attn_norm_conv_env_hpf_data = norm_conv_env_hpf_data / attn
        out_data[rec, :] = attn_norm_conv_env_hpf_data

        # test output
        if flg_test_graph_out:
            fig_test, ax1 = plt.subplots(
                figsize=(6.4, 4.8))  # default dpi = 100
            #ax1.plot(x, multi_data[0])
            #ax1.plot(x, conv_data)
            #ax1.plot(x, hpf_data)
            #ax1.plot(x, abs_data)
            #ax1.plot(x, conv_data_2)
            #ax1.plot(x, envelope)
            #ax1.plot(x, conv_data_3)
            ax1.plot(x, norm_conv_env_hpf_data)
            #ax2 = ax1.twinx()
            ax1.plot(x, attn_norm_conv_env_hpf_data)
            #ax2.plot(x, multi_data[0])
            ax1.set_xlabel("Depth [mm]")
            ax1.set_ylabel("Intensity [arb. Unit]")
            fig_test.show()
            #messagebox.showinfo('Information', "Execution has done")
    # draw movie file
    path_video = os.path.join(path, 'video_out.mp4')
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(path_video, four_cc, 8.0, (640, 480))
    for frame in tqdm(range(num_rec), leave=False, desc="Drawing graph plot..."):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
        ax.plot(x, out_data[frame, :])
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Intensity [arb. Unit]")
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        video.write(img)
        plt.close()
    video.release
