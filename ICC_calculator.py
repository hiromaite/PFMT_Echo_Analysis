import os
import tkinter
from tkinter import filedialog, messagebox
from tkinter.constants import *

import cv2
from scipy import signal
from scipy.stats import norm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants for m-mode slicer
cs_center = 150
mean_width = 16

# Open file select dialog, check file exist
root = tkinter.Tk()
root.withdraw()
fTyp = [('png file', '*.png')]
iDir = os.path.abspath(os.path.dirname(__file__))
files = filedialog.askopenfilenames(filetypes=fTyp,
                                    initialdir=iDir)  # video file(s)
try:
    stat = os.stat(files[0])  # do try-catch with first file
except:
    messagebox.showinfo('Information',
                        "Execution has canceled in dialog window")
    exit()

for file in files:
    # get file data, param.
    path = os.path.dirname(file)
    filename = os.path.splitext(os.path.basename(file))[0]
    cap = cv2.VideoCapture(file)  # video object

    # Set variables about video object
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("読み込まれたファイルの絶対パスは{}，".format(path))
    print("ファイル名は{}，".format(filename))
    print("画像の解像度は，幅{}，高さ{}です。".format(frame_width, frame_height))