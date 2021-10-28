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


def draw_line(event, x, y, flags, param):
    # callback function when a mouse event occurs
    global x_1, y_1, x_2, y_2, fl_first, fl_second, fl_cmpl

    if event == cv2.EVENT_LBUTTONDOWN:
        if not(fl_first):
            x_1, y_1 = x, y
            cv2.circle(img_draw, (x_1, y_1), 2, (0, 0, 255), thickness=1)
            fl_first = True
        elif not(fl_second):
            x_2, y_2 = x, y
            cv2.circle(img_draw, (x_2, y_2), 2, (0, 0, 255), thickness=1)
            fl_second = True
        elif fl_first and fl_second:
            fl_cmpl = True


# Constants for m-mode slicer
cs_center = 150
mean_width = 16

# Open file select dialog, check file exist
root = tkinter.Tk()
root.withdraw()
fTyp = [('jpg file', '*.jpg')]
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
    img = cv2.imread(file)  # image object

    #  Set variables about video object
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1

    print("読み込まれたファイルの絶対パスは'{}'，".format(path))
    print("ファイル名は'{}'，".format(filename))
    print("画像の解像度は，幅:{}，高さ:{}です。".format(width, height))

    # get the coordinates of upper and bottom of the bladder for rotation
    fl_first = False
    fl_second = False
    fl_cmpl = False
    x_1, y_1 = -1, -1
    x_2, y_2 = -1, -1
    img_draw = img
    cv2.namedWindow(winname='my_drawing')
    cv2.setMouseCallback('my_drawing', draw_line)
    while True:
        cv2.imshow('my_drawing', img_draw)
        if (cv2.waitKey(1) & 0xFF == 27) or fl_cmpl:
            break
    cv2.destroyAllWindows()

    if not(fl_cmpl):
        break
    
    # get center point of rotate
    x_center = (x_1 + x_2) / 2
    y_center = (y_1 + y_2) / 2

    # get rotate angle
    rotate_angle = 90 - np.degrees(np.arctan(abs(y_2 - y_1)/abs(x_2 - x_1)))
    sign = np.sign((y_2 - y_1)/(x_2 - x_1))
    rotate_angle = -1 * sign * rotate_angle

    # rotate image and wrap image
    trans = cv2.getRotationMatrix2D((x_center, y_center), rotate_angle, 1)
    img_rotate_affine = cv2.warpAffine(img, trans, (width,height),flags=cv2.INTER_CUBIC)

    print("膀胱前壁は，x = {}，y = {}，".format(x_1, y_1))
    print("膀胱底は，x = {}，y = {}，".format(x_2, y_2))
    print("Mモード取得断面の中心は，x = {}です。".format(x_center))

    while True:
        cv2.imshow('result', img_rotate_affine)
        if cv2.waitKey(1) & 0xFF == 27:
            break