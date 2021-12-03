import tkinter
from tkinter import filedialog, messagebox
import os

import numpy as np
import scipy.io
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

flg_test_graph_out = False

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

    # output data
    out_bottom = []
    out_top = []

    # time scale
    x = np.linspace(0, rec_len - 1, rec_len)
    x = x - 0.1 * rec_len  # trigger position
    x = x * (20 * 0.001 * 0.001) / 1250  # sampling rate (=20us)
    x = x * 1520 * 1000  # speed of sound

    # attenuation coefficient
    db = (x / 10) * -1
    attn = 10 ** (db / 20)

    # Gaussian function
    x_gauss = np.arange(-10, 10, 20 / 1000)
    weight = norm.pdf(x_gauss)

    for rec in tqdm(range(num_rec), leave=False):
        # get the convolution integral with the Gaussian function
        attnd_data = multi_data[rec] / attn
        conv_data = np.convolve(attnd_data, weight, mode='same') / np.sum(weight)

        # origin correction and normalization based on min and max
        max_conv_data = max(conv_data[15:-15])
        min_conv_data = min(conv_data[15:-15])
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
        out_top.append(- depth_top)

        # test output
        if flg_test_graph_out and rec == 0:
            fig_test, ax1 = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
            ax1.plot(x, attnd_data)
            ax1.plot(x, conv_data)
            ax2 = ax1.twinx()
            ax2.plot(x, conv_data_norm)
            ax2.plot(x[:-1], silent_area)
            ax2.plot(x[:-1], mass_dist_data)
            #ax.plot(x, multi_data[0])
            # ax.set_ylim(0, 180)
            ax1.set_xlabel("Depth [mm]")
            ax1.set_ylabel("Intensity [arb. Unit]")
            fig_test.show()
            messagebox.showinfo('Information', "Execution has done")
    
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