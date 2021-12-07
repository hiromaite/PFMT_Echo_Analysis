import tkinter
from tkinter import Label, filedialog, messagebox
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
flg_graph_out = False
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

    # output data
    out_bottom = []
    out_top = []
    out_data = np.zeros((num_rec, rec_len), float)

    # Gaussian function
    x_gauss = np.arange(-4, 4, 8 / 1000)
    weight = norm.pdf(x_gauss)

    for rec in tqdm(range(num_rec), leave=False):
        # get the convolution integral with the Gaussian function
        freq = 2 * 1000 * 1000
        conv_data = gaussian_filter(multi_data[rec], sample_rate, freq)

        # high-pass filter
        fp = 2 * 1000 * 1000
        fs = 1 * 1000 * 1000
        hpf_data = highpass_filter(conv_data, sample_rate, fp, fs)

        # get envelope wave
        env_hpf_data = abs(signal.hilbert(hpf_data))

        # get the convolution integral with the Gaussian function
        freq = 0.2 * 1000 * 1000
        conv_env_hpf_data = gaussian_filter(env_hpf_data, sample_rate, freq)

        # normalized by min & max value while region of Rx signals from body
        conv_env_hpf_data[0: int(0.2 * rec_len)] = 0
        conv_env_hpf_data[int(-0.05 * rec_len):] = 0
        region_start = int(0.4 * rec_len)
        region_end = -1
        min_data = min(conv_env_hpf_data[region_start: region_end])
        max_data = max(conv_env_hpf_data[region_start: region_end])
        norm_conv_env_hpf_data = (
            conv_env_hpf_data - min_data) / (max_data - min_data)

        # get integrated values
        int_data = conv_env_hpf_data.copy()
        for i in range(rec_len - 1):
            int_data[i + 1] = int_data[i] + conv_env_hpf_data[i + 1]
        int_data = int_data / int_data[-1]

        # make attenuation function
        db = (x / 10) * -1
        db = np.multiply(db, int_data)
        attn = 10 ** (db / 20)
        out_raw = conv_env_hpf_data / attn
        #attn_norm_conv_env_hpf_data = np.multiply(conv_env_hpf_data, int_data**12) - 0.00005
        #attn_norm_conv_env_hpf_data = np.where(attn_norm_conv_env_hpf_data < 0, 0, attn_norm_conv_env_hpf_data)
        #attn_norm_conv_env_hpf_data = np.where(attn_norm_conv_env_hpf_data >= 0.0003, 0.0003, attn_norm_conv_env_hpf_data)
        #out_raw = attn_norm_conv_env_hpf_data / 0.0003
        out_data[rec, :] = out_raw

        conv_data = np.convolve(out_raw, weight, mode='same') / np.sum(weight)

        # origin correction and normalization based on min and max
        max_conv_data = max(conv_data[int(0.3 * rec_len): int(-0.1 * rec_len)])
        min_conv_data = min(conv_data[int(0.3 * rec_len): int(-0.1 * rec_len)])
        conv_data_norm = ((conv_data - min_conv_data)
                        / (max_conv_data - min_conv_data))

        # get the differential value
        diff = 1  # order(s)
        diff_data = np.diff(conv_data_norm, diff)

        # separate the silent area(background) from the sound area
        silent_area = np.where((-0.02 < diff_data) & (diff_data < 0.02)
                            & (conv_data_norm[: -1] < 0.1), 0, 0.05)

        # get the integral value for each sound area
        int_int_data = silent_area.copy()  # var for interval integral
        for i in range(len(int_int_data) - 1):
            if silent_area[i + 1]:  # 0 means False
                int_int_data[i + 1] = int_int_data[i] + silent_area[i + 1]

        # give the integral value to the sound area
        mass_dist_data = int_int_data
        for i in range(len(mass_dist_data) - 1)[::-1]:
            if int_int_data[i - 1] and int_int_data[i - 1] < int_int_data[i]:
                mass_dist_data[i - 1] = int_int_data[i]
        mass_dist_data[-1] = 0
        mass_dist_data_all = mass_dist_data.copy()

        # normalized by max intensity after removing the area of multiple intensities
        mass_dist_data_top = np.zeros(len(mass_dist_data), np.float16)
        for i in range(len(mass_dist_data)):
            if not(mass_dist_data[i]):
                break
            mass_dist_data[i] = 0
        max_mass_data = max(mass_dist_data)
        mass_dist_data = mass_dist_data / (10 * max_mass_data)
        mass_dist_data_all = mass_dist_data_all / (10 * max_mass_data)

        # find the highest and second highest masses(=peak group or sound area)
        mass_first = -1
        mass_second = -1
        for i in range(len(mass_dist_data)):
            if mass_dist_data[i] > mass_first:
                mass_second = mass_first
                mass_first = mass_dist_data[i]
            elif mass_dist_data[i] == mass_first:
                pass
            elif mass_dist_data[i] > mass_second:
                mass_second = mass_dist_data[i]
        mass_dist_data = np.where((mass_dist_data == mass_first)
                                | (mass_dist_data == mass_second), mass_dist_data, 0)

        # get the range of the mass region(s) from the shallow side
        id_shallow_start = -1
        id_shallow_end = -1
        id_deep_start = -1
        id_deep_end = -1
        for i in range(len(mass_dist_data) - 1):
            if mass_dist_data[i + 1] > mass_dist_data[i]:
                if id_shallow_start == -1:
                    id_shallow_start = i + 1
                else:
                    id_deep_start = i + 1
            elif mass_dist_data[i + 1] < mass_dist_data[i]:
                if id_shallow_end == -1:
                    id_shallow_end = i + 1
                else:
                    id_deep_end = i + 1

        # select peak and define depth of bladder bottom
        cpd = 1520 * 100 * (20 * 0.001 * 0.001) / 1250  # centimeter per dot
        if id_deep_start == -1:  # when not detected double peak
            id_bottom = id_shallow_start
        elif mass_dist_data[id_shallow_start] > mass_dist_data[id_deep_start]:
            id_bottom = id_shallow_start
        elif (id_shallow_end - id_shallow_start) / 2 < (id_deep_start - id_shallow_end):
            id_bottom = id_deep_start
        else:
            id_bottom = id_shallow_start
        depth_bottom = (id_bottom - 1250) * cpd
        if depth_bottom < 5 and len(out_bottom) > 0:
            depth_bottom = - out_bottom[-1]
        out_bottom.append(- depth_bottom)

        # select peak and define depth of bladder top
        id_top = -1
        for i in range(1, id_bottom - 1)[::-1]:
            if mass_dist_data_all[i] - mass_dist_data_all[i - 1] > 0:
                break
            mass_dist_data_top[i] = mass_dist_data_all[i]
        for i in range(len(mass_dist_data_top) - 1):
            if mass_dist_data_top[i + 1] < mass_dist_data_top[i]:
                id_top = i
                break
        depth_top = (id_top - 1250) * cpd
        if depth_top < 5 and len(out_top) > 0:
            depth_top = - out_top[-1]
        if len(out_top) > 1 and depth_top > -1 * max(out_bottom[1: -1]):
            depth_top = - out_top[-1]
        out_top.append(- depth_top)

        # test output
        if flg_test_graph_out and rec == 0:
            fig_test, ax1 = plt.subplots(
                figsize=(6.4, 4.8))  # default dpi = 100
            #ax1.plot(x, multi_data[rec], label='RAW data')
            #ax1.plot(x, conv_data, label='Low pass filtered')
            #ax1.plot(x, hpf_data, label='High pass filtered')
            #ax1.plot(x, env_hpf_data, label='Envelope')
            #ax1.plot(x, conv_env_hpf_data, label='Convolved w/ Gaussian(200kHz width)')
            ax1.plot(x, conv_env_hpf_data, label='Convolved w/ Gaussian(200kHz width)')
            ax2 = ax1.twinx()
            ax2.plot(x, int_data, color='C1', label='Integral of intensity')
            ax3 = ax1.twinx()
            #ax3.plot(x, attn, color='C2', label='Attenuation function')
            ax3.plot(x, attn_norm_conv_env_hpf_data, color='C3', label='Normalized by attenuation function')
            ax1.set_xlabel("Depth [mm]")
            ax1.set_ylabel("Intensity [arb. Unit]")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            ax1.legend(h1 + h2 + h3, l1 + l2 + l3)
            fig_test.show()
            messagebox.showinfo('Information', "Execution has done")
            exit()

    #out_bottom = gaussian_filter(out_bottom, 10, 1)
    #out_top = gaussian_filter(out_top, 10, 1)

    # graph out
    t = np.linspace(0, 24, num_rec)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(t, out_bottom, label='Bladder bottom')
    ax.plot(t, out_top, label='Bladder top')
    ax.set_xlabel("Meas. time [sec]")
    ax.set_ylabel("Depth [mm]")
    plt.legend()
    path_figure = os.path.join(path, filename + '.png')
    fig.savefig(path_figure)
    plt.clf
    plt.close()

    # draw movie file
    if flg_graph_out:
        path_video = os.path.join(path, 'video_out.mp4')
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(path_video, four_cc, 8.0, (640, 480))
        for frame in tqdm(range(num_rec), leave=False, desc="Drawing graph plot..."):
            fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
            ax.plot(x, out_data[frame, :])
            #ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Depth [mm]")
            ax.set_ylabel("Intensity [arb. Unit]")
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            video.write(img)
            plt.close()
        video.release
