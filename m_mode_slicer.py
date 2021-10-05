import os
import tkinter
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from tqdm import tqdm


def m_mode(cap, frame_count, frame_width, frame_height, cs_center, mean_width):
    # convert echo movie into M-mode picture by averaging with 'mean_width' at 'cs_center'
    m_mode_array = np.zeros((frame_height, frame_count), np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_count):
        _, frame = cap.read()
        colmun = np.zeros((frame_height, 3), np.uint8)
        for j in range(mean_width):
            # calc. average in each rows(= num of mean_width)
            colmun[:, j] = frame[:, cs_center - int(mean_width / 2) + j, 0]
        merge = np.average(colmun, axis=1)
        m_mode_array[:, i] = merge
    return m_mode_array


# Open file select dialog, check file exist
root = tkinter.Tk()
root.withdraw()
fTyp = [('mp4 file', '*.mp4')]
iDir = os.path.abspath(os.path.dirname(__file__))
files = filedialog.askopenfilenames(
    filetypes=fTyp, initialdir=iDir)  # video file(s)
try:
    stat = os.stat(files[0])  # do try-catch with first file
except:
    messagebox.showinfo(
        'Information', "Execution has canceled in dialog window")
    exit()

for file in tqdm(files):
    path = os.path.dirname(file)
    filename = os.path.splitext(os.path.basename(file))[0]
    cap = cv2.VideoCapture(file)  # video object

    # Set variables about video object
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # convert echo into m-mode picture
    m_mode_array = m_mode(cap, frame_count, frame_width, frame_height, 150, 3)

    #cv2.imshow("Execution Result", m_mode_array)
    cv2.imwrite(os.path.join(path, filename + '_mMode.jpg'), m_mode_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    cap.release()

messagebox.showinfo('Done!', "All Echo videos have been converted.")
