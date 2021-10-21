import os
import tkinter
from tkinter import StringVar, filedialog, messagebox, ttk
from tkinter.constants import *
from PIL import Image, ImageTk

import cv2
from scipy import signal
from scipy.stats import norm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def m_mode(cap, frame_count, frame_width, frame_height, cs_center, mean_width):
    # convert echo movie into M-mode picture by averaging with 'mean_width' at 'cs_center'
    m_mode_array = np.zeros((frame_height, frame_count, mean_width), np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_count):
        _, frame = cap.read()
        colmun = np.zeros((frame_height, mean_width), np.uint8)
        for j in range(mean_width):
            # calc. average in each rows(= num of mean_width)
            colmun[:, j] = frame[:, cs_center - int(mean_width / 2) + j, 0]
        merge = np.average(colmun, axis=1)
        m_mode_array[:, i, 0] = merge
        m_mode_array[:, i, 1] = merge
        m_mode_array[:, i, 2] = merge
    return m_mode_array


# Constants for m-mode slicer
cs_center = 150
mean_width = 16

# Open file select dialog, check file exist
root = tkinter.Tk()
root.withdraw()
fTyp = [('mp4 file', '*.mp4')]
iDir = os.path.abspath(os.path.dirname(__file__))
files = filedialog.askopenfilenames(filetypes=fTyp,
                                    initialdir=iDir)  # video file(s)
try:
    stat = os.stat(files[0])  # do try-catch with first file
except:
    messagebox.showinfo('Information',
                        "Execution has canceled in dialog window")
    exit()

# Execution portion
for file in files:
    path = os.path.dirname(file)
    filename = os.path.splitext(os.path.basename(file))[0]
    cap = cv2.VideoCapture(file)  # video object

    # Set variables about video object
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # convert echo into m-mode picture
    m_mode_array = m_mode(cap, frame_count, frame_width,
                          frame_height, int(cs_center), int(mean_width))

    # algorithm portion
    out_data = np.zeros((frame_height, frame_count), np.float32)
    maxid = [0] * frame_count
    minid = [0] * frame_count
    x_gauss = np.arange(-5, 5, 10/100)
    weight = norm.pdf(x_gauss)
    diff = 1  # orders of differentiation
    for i in range(frame_count):
        out_data[:, i] = np.convolve(
            m_mode_array[:, i, 0], weight, mode='same')/np.sum(weight)
        out_data[: -1*diff, i] = np.diff(out_data[:, i], diff)
        maxid[i] = signal.argrelmax(out_data[: -1*diff, i], order=10)
        minid[i] = signal.argrelmin(out_data[: -1*diff, i], order=10)

    # make video file from plot image, save init path
    path_out = os.path.join(path, filename + '_DIFF' + '.mp4')
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(path_out, four_cc, 8.0, (640, 480))

    x = np.arange(0, 18, 18/300)
    for i in tqdm(range(frame_count), leave=False):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
        # ax.plot(x[: -1*diff], out_data[: -1*diff, i])
        ax.plot(x[:], m_mode_array[:, i, 0])
        # ax.plot(x[maxid[i]], np.squeeze(out_data[maxid[i], i]), 'ro')
        # ax.plot(x[minid[i]], np.squeeze(out_data[minid[i], i]), 'bo')
        ax.plot(x[maxid[i]], np.squeeze(m_mode_array[maxid[i], i, 0]), 'ro')
        ax.plot(x[minid[i]], np.squeeze(m_mode_array[minid[i], i, 0]), 'bo')
        ax.set_ylim(0, 180)
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        video.write(img)
        plt.close()

    video.release
