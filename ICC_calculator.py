import os
import tkinter
from tkinter import filedialog, messagebox
from tkinter.constants import *
import csv

import cv2
import numpy as np
import scipy
from scipy import signal
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

flg_rot = False  # flag to enable rotation function
flg_test = False  # for review
flg_figure_out = True  # for review

# For religious reasons, the direction of the tick marks on the diagram should be inward.
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def draw_line(event, x, y, flags, param):
    # callback function when a mouse event occurs
    global x_1, y_1, x_2, y_2, flg_first, flg_second, flg_cmpl

    if event == cv2.EVENT_LBUTTONDOWN:
        if not(flg_first):
            x_1, y_1 = x, y
            cv2.circle(img_draw, (x_1, y_1), 2, (0, 0, 255), thickness=1)
            flg_first = True
        elif not(flg_second):
            x_2, y_2 = x, y
            cv2.circle(img_draw, (x_2, y_2), 2, (0, 0, 255), thickness=1)
            flg_second = True
        elif flg_first and flg_second:
            flg_cmpl = True


def rot_img(img):
    # get the coordinates of upper and bottom of the bladder for rotation
    global x_1, y_1, x_2, y_2, flg_first, flg_second, flg_cmpl

    flg_first = False
    flg_second = False
    flg_cmpl = False
    x_1, y_1 = -1, -1
    x_2, y_2 = -1, -1
    cv2.namedWindow(winname='my_drawing')
    cv2.setMouseCallback('my_drawing', draw_line)
    while True:
        cv2.imshow('my_drawing', img_draw)
        if (cv2.waitKey(1) & 0xFF == 27) or flg_cmpl:
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
    if flg_test:
        print("膀胱前壁は，x = {}，y = {}，".format(x_1, y_1))
        print("膀胱底は，x = {}，y = {}，".format(x_2, y_2))
        print("Mモード取得断面の中心は，x = {}です。".format(x_center))
        while True:
            cv2.imshow('result', img_draw_rotate)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        return img_rot, x_center


# Constants for m-mode slicer
cs_center = 150  # center of cross-section
mean_width = 16  # diameter of ultra-sound wave beam

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
                        "Execution has canceled in dialog window.")
    exit()
if len(files) % 2 != 0:
    messagebox.showerror('Error!',
                         "Select an even number of files.")
    exit()

# (option) make list to store the output data
out = [[] for i in range(len(files) // 2)]

# iteration per file
for counter, file in enumerate(tqdm(files)):
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
    if flg_test:
        print("読み込まれたファイルの絶対パスは'{}'，".format(path))
        print("ファイル名は'{}'，".format(filename))
        print("画像の解像度は，幅:{}，高さ:{}です。".format(width, height))

    # echo image rotation function, Not used at the start of PoC
    if flg_rot:
        img, cs_center = rot_img(img)
        if not(flg_cmpl):
            break

    # convolve echo image to one-D data
    one_dim_data = np.zeros((height), np.uint8)
    colmun = np.zeros((height, mean_width), np.uint8)
    for i in range(mean_width):
        # calc. average in each rows(= num of mean_width)
        colmun[:, i] = img[:, int(cs_center - (mean_width / 2) + i), 0]
    one_dim_data = np.average(colmun, axis=1)

    # get the convolution integral with the Gaussian function
    x_gauss = np.arange(-10, 10, 20 / 100)
    weight = norm.pdf(x_gauss)
    conv_data = np.convolve(
        one_dim_data, weight, mode='same') / np.sum(weight)

    # origin correction and normalization based on min and max
    max_conv_data = max(conv_data[15:-15])
    min_conv_data = min(conv_data[:165])
    conv_data_norm = ((conv_data - min_conv_data)
                      / (max_conv_data - min_conv_data))

    # get the differential value
    diff = 1  # order(s)
    diff_data = np.diff(conv_data_norm, diff)

    # separate the silent area(background) from the sound area
    silent_area = np.where((-0.02 < diff_data) & (diff_data < 0.02)
                           & (conv_data_norm[: -1] < 0.1), 0, 0.05)

    # get the integral value for each sound area
    int_int_data = silent_area  # var for interval integral
    for i in range(len(int_int_data) - 1):
        if silent_area[i + 1]:  # 0 means False
            int_int_data[i + 1] = int_int_data[i] + silent_area[i + 1]
        mass_dist_data = int_int_data
    for i in range(len(mass_dist_data) - 1)[::-1]:
        if int_int_data[i - 1] and int_int_data[i - 1] < int_int_data[i]:
            mass_dist_data[i - 1] = int_int_data[i]
    mass_dist_data[-1] = 0
    mass_dist_data_all = mass_dist_data.copy()
    mass_dist_data_top = np.zeros(len(mass_dist_data), np.float16)
    for i in range(len(mass_dist_data)):
        if not(mass_dist_data[i]):
            break
        mass_dist_data[i] = 0
    max_mass_data = max(mass_dist_data)
    mass_dist_data = mass_dist_data / (10 * max_mass_data)
    mass_dist_data_all = mass_dist_data_all / (10 * max_mass_data)

    # find the highest and second highest masses(=peak group)
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
    cpd = 17 / 300  # centimeter per dot
    if id_deep_start == -1:  # when not detected double peak
        id_bottom = id_shallow_start
    elif mass_dist_data[id_shallow_start] > mass_dist_data[id_deep_start]:
        id_bottom = id_shallow_start
    elif (id_shallow_end - id_shallow_start) / 2 < (id_deep_start - id_shallow_end):
        id_bottom = id_deep_start
    else:
        id_bottom = id_shallow_start
    depth_bottom = id_bottom * cpd

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
    depth_top = id_top * cpd

    # append data into output
    pair = counter // 2
    if counter % 2 == 0:  # Processing of even-numbered files
        out[pair] = [filename, depth_bottom, depth_bottom - depth_top]
    else:
        out[pair].extend([filename, depth_bottom, depth_bottom - depth_top,
                         out[pair][1] - depth_bottom, out[pair][2] - (depth_bottom - depth_top)])

    # test output
    if flg_test:
        x = np.arange(0, 17, 17/300)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
        ax.plot(x, one_dim_data)
        ax.set_ylim(0, 180)
        fig.show()
        messagebox.showinfo('Information', "Execution has done")

    # output processing results as a graph
    if flg_figure_out:
        x = np.arange(0, 17, 17/300)
        fig_out, axes = plt.subplots(
            nrows=2, ncols=1, squeeze=False, sharex='all', figsize=(6.4, 4.8))  # default dpi = 180

        axes[0, 0].plot(x, one_dim_data)
        axes[0, 0].plot(x, conv_data)
        axes[0, 0].set_xlim(0, 17)
        axes[0, 0].set_ylim(0, 180)
        axes[0, 0].set_ylabel("Intensity")

        axes[1, 0].plot(x[: -1], diff_data)
        axes[1, 0].plot(x[: -1], mass_dist_data)
        axes[1, 0].plot(x[: -1], mass_dist_data_top)
        axes[1, 0].set_ylim(-0.1, 0.1)
        axes[1, 0].set_xlabel("Depth")
        axes[1, 0].set_ylabel("Difference")

        fig_out.align_labels(axes)

        path_figure = os.path.join(path + '/figure', filename + '.png')
        fig_out.savefig(path_figure)
        plt.clf
        plt.close()

# test output
if flg_test:
    print("{} files are converted into lifting height.".format(counter))
    print("For example, first data are below...")
    print(out[0])

# output lifting height as a CSV file
path_out = os.path.join(path, 'lifting_height_results.csv')
with open(path_out, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(out)
