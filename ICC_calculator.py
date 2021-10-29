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

fl_rot = False  # flag to enable rotation function


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


def rot_img(img):
    # get the coordinates of upper and bottom of the bladder for rotation
    global x_1, y_1, x_2, y_2, fl_first, fl_second, fl_cmpl

    fl_first = False
    fl_second = False
    fl_cmpl = False
    x_1, y_1 = -1, -1
    x_2, y_2 = -1, -1
    cv2.namedWindow(winname='my_drawing')
    cv2.setMouseCallback('my_drawing', draw_line)
    while True:
        cv2.imshow('my_drawing', img_draw)
        if (cv2.waitKey(1) & 0xFF == 27) or fl_cmpl:
            break
    cv2.destroyAllWindows()

    # get center point of rotate
    """
    Rotate at the midpoint between the anterior bladder wall
    and the bladder base to avoid loss of information that can be used in the algorithm.
    """
    x_center = (x_1 + x_2) / 2
    y_center = (y_1 + y_2) / 2

    # get rotate angle
    """
    Rotate the image so that the two selected points on the echo image are vertical.
    For this purpose, the angle between the line segment created by the two points
    and the vertical is calculated and stored in var:"rotate_angle".
    Furthermore, it determines whether the rotation is clockwise or counterclockwise
    relative to the vertical and stores it in var:"sign".
    """
    rotate_angle = 90 - np.degrees(np.arctan(abs(y_2 - y_1)/abs(x_2 - x_1)))
    sign = np.sign((y_2 - y_1)/(x_2 - x_1))  # CW?(-) or CCW?(+)
    rotate_angle = -1 * sign * rotate_angle

    # rotate image and wrap image
    trans = cv2.getRotationMatrix2D((x_center, y_center), rotate_angle, 1)
    img_rot = cv2.warpAffine(
        img, trans, (width, height), flags=cv2.INTER_CUBIC)
    img_draw_rotate = cv2.warpAffine(
        img_draw, trans, (width, height), flags=cv2.INTER_CUBIC)

    # test output
    print("膀胱前壁は，x = {}，y = {}，".format(x_1, y_1))
    print("膀胱底は，x = {}，y = {}，".format(x_2, y_2))
    print("Mモード取得断面の中心は，x = {}です。".format(x_center))
    while True:
        cv2.imshow('result', img_draw_rotate)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    return img_rot, x_center


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
    img_draw = cv2.imread(file)  # image object drawing use

    #  Set variables about video object
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1

    # test output
    print("読み込まれたファイルの絶対パスは'{}'，".format(path))
    print("ファイル名は'{}'，".format(filename))
    print("画像の解像度は，幅:{}，高さ:{}です。".format(width, height))

    if fl_rot:
        img = rot_img(img)
        if not(fl_cmpl):
            break

    # convolve echo data
    m_mode_array = np.zeros((height), np.uint8)
    colmun = np.zeros((height, mean_width), np.uint8)
    for i in range(mean_width):
        # calc. average in each rows(= num of mean_width)
        colmun[:, i] = img[:, int(cs_center - (mean_width / 2) + i), 0]
    merge = np.average(colmun, axis=1)
    m_mode_array = merge

    # run algorizm
    out_data = np.zeros((height), np.float32)
    max_id = []
    min_id = []
    x_gauss = np.arange(-5, 5, 10/100)
    weight = norm.pdf(x_gauss)
    diff = 1  # orders of differentiation
    out_data = np.convolve(
        m_mode_array[:], weight, mode='same')/np.sum(weight)
    out_data[: -1*diff] = np.diff(out_data[:], diff)
    max_id = np.squeeze(signal.argrelmax(out_data[: -1*diff], order=10))
    del_id = []
    id = 0
    for i in max_id:
        a = out_data[i]
        if out_data[i] < 1 and out_data[i] > -1:
            del_id.append(id)
        id += 1
    temp_id = max_id
    for r in del_id[::-1]:
        temp_id = np.delete(temp_id, r, 0)
        max_id = temp_id
    min_id = np.squeeze(signal.argrelmin(out_data[: -1*diff], order=10))
    del_id = []
    id = 0
    for i in min_id:
        if out_data[i] < 1 and out_data[i] > -1:
            del_id.append(id)
        id += 1
    temp_id = min_id
    for r in del_id[::-1]:
        temp_id = np.delete(temp_id, r, 0)
        min_id = temp_id

    x = np.arange(0, 18, 18/300)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
    ax.plot(x, m_mode_array)
    ax.plot(x[max_id], m_mode_array[max_id], 'ro')
    ax.plot(x[min_id], m_mode_array[min_id], 'bo')
    ax.set_ylim(0, 180)
    fig.show()

    messagebox.showinfo('Information', "Execution has done")
