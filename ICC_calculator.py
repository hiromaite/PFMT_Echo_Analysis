import os
import tkinter
from tkinter import filedialog, messagebox
from tkinter.constants import *
import csv

import cv2
from scipy import signal
from scipy.stats import norm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

flg_rot = False  # flag to enable rotation function
flg_test = False  # for review
flg_figure_out = True  # for review


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
    merge = np.average(colmun, axis=1)
    one_dim_data = merge

    # run algorizm
    max_id = []
    min_id = []
    x_gauss = np.arange(-5, 5, 10/100)
    weight = norm.pdf(x_gauss)
    diff = 1  # orders of differentiation
    conv_data = np.convolve(
        one_dim_data[:], weight, mode='same')/np.sum(weight)
    diff_data = np.diff(conv_data[:], diff)
    max_id = np.squeeze(signal.argrelmax(diff_data[: -1*diff], order=10))
    del_id = []
    id = 0
    for i in max_id:
        if i < 50 and 280 < i:
            del_id.append(id)
        elif -1 < diff_data[i] < 1:
            del_id.append(id)
        id += 1
    temp_id = max_id
    for r in del_id[::-1]:
        temp_id = np.delete(temp_id, r, 0)
        max_id = temp_id
    min_id = np.squeeze(signal.argrelmin(diff_data[: -1*diff], order=10))
    del_id = []
    id = 0
    for i in min_id:
        if i < 50 or 280 < i:
            del_id.append(id)
        elif -1 < diff_data[i] < 1:
            del_id.append(id)
        id += 1
    temp_id = min_id
    for r in del_id[::-1]:
        temp_id = np.delete(temp_id, r, 0)
        min_id = temp_id

    # test output
    if flg_test:
        x = np.arange(0, 18, 18/300)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
        ax.plot(x, one_dim_data)
        ax.plot(x[max_id], one_dim_data[max_id], 'ro')
        ax.plot(x[min_id], one_dim_data[min_id], 'bo')
        ax.set_ylim(0, 180)
        fig.show()
        messagebox.showinfo('Information', "Execution has done")

    if flg_figure_out:
        x = np.arange(0, 18, 18/300)
        fig_out = plt.figure(figsize=(6.4, 4.8))  # default dpi = 180

        ax1 = fig_out.add_subplot(2, 1, 1)
        ax1.plot(x, one_dim_data)
        ax1.plot(x, conv_data)
        ax1.plot(x[max_id], one_dim_data[max_id], 'ro')
        ax1.plot(x[min_id], one_dim_data[min_id], 'bo')
        ax1.set_ylim(0, 180)
        ax1.set_ylabel("Intensity")

        ax2 = fig_out.add_subplot(2, 1, 2)
        ax2.plot(x[: -1], diff_data)
        ax2.set_ylim(-4, 4)
        ax2.set_xlabel("Depth")
        ax2.set_ylabel("Difference")

        path_figure = os.path.join(path + '/figure', filename + '.png')
        fig_out.savefig(path_figure)
        plt.clf
        plt.close()

    # append data into output
    """
    If no peak is found, treat as 'N/A'.
    Even if there is a peak at the time of elevation,
    if there is no peak at the time of relaxation,
    it is treated as 'N/A' as the amount of elevation.
    """
    pair = counter // 2
    if counter % 2 == 0:  # Processing of even-numbered files
        if len(max_id) == 0:  # peak number is 0
            out[pair] = [filename, 'none', 'N/A', 'N/A']
        elif len(max_id) == 1:  # peak number is 1
            depth_peaks = max_id[0] * (17 / 300)
            out[pair] = [filename, 'single', depth_peaks, 'N/A']
        else:  # peak number is 2 or higher
            depth_peaks = [max_id[0] * (17 / 300), max_id[1] * (17 / 300)]
            out[pair] = [filename, 'double', depth_peaks[0], depth_peaks[1]]
        if len(min_id) == 0:
            out[pair].extend(['N/A'])
        else:
            depth_valley = min_id[0] * (17 / 300)
            out[pair].extend([depth_valley])
    else:  # Processing of odd-numbered files
        if len(max_id) == 0:
            out[pair].extend([filename, 'none', 'N/A', 'N/A'])
        elif len(max_id) == 1:
            depth_peaks = max_id[0] * (17 / 300)
            out[pair].extend([filename, 'single', depth_peaks, 'N/A'])
        else:
            depth_peaks = [max_id[0] * (17 / 300), max_id[1] * (17 / 300)]
            out[pair].extend([filename, 'double', depth_peaks[0], depth_peaks[1]])
        if len(min_id) == 0:
            out[pair].extend(['N/A'])
        else:
            depth_valley = min_id[0] * (17 / 300)
            out[pair].extend([depth_valley])

        # Calculated from peak group
        if out[pair][1] == 'none' or out[pair][6] == 'none':
            out[pair].extend(['N/A'])
        elif out[pair][1] == 'double' and out[pair][6] == 'double':
            lifting_hight = [out[pair][2] - out[pair][7], out[pair][3] - out[pair][8]]
            out[pair].extend([max(lifting_hight)])
        elif out[pair][1] == 'single' and out[pair][6] == 'single':
            lifting_hight = out[pair][2] - out[pair][7]
            out[pair].extend([lifting_hight])
        elif out[pair][1] == 'double' and out[pair][6] == 'single':
            lifting_hight = [out[pair][2] - out[pair][7], out[pair][3] - out[pair][7]]
            out[pair].extend([max(lifting_hight)])
        elif out[pair][1] == 'single' and out[pair][6] == 'double':
            lifting_hight = [out[pair][2] - out[pair][7], out[pair][2] - out[pair][8]]
            out[pair].extend([max(lifting_hight)])

        # Calculated from the end position of the pelvic floor muscle group
        if out[pair][4] == 'N/A' or out[pair][9] == 'N/A':
            out[pair].extend(['N/A'])
        else:
            lifting_hight = out[pair][4] - out[pair][9]
            out[pair].extend([lifting_hight])
        
        # Select a plausible value
        if out[pair][11] == 'N/A':
            out[pair].extend([out[pair][10]])
        else:
            if out[pair][11] > 2 * out[pair][10]:
                out[pair].extend([out[pair][11]])
            elif out[pair][10] > 1.5 * out[pair][11]:
                out[pair].extend([out[pair][11]])
            else:
                out[pair].extend([out[pair][10]])

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
