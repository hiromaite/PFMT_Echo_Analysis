import os
import tkinter
from tkinter import StringVar, filedialog, messagebox, ttk
from tkinter.constants import *
from PIL import Image, ImageTk

import cv2
import numpy as np
from tqdm import tqdm


def m_mode(cap, frame_count, frame_width, frame_height, cs_center, mean_width):
    # convert echo movie into M-mode picture by averaging with 'mean_width' at 'cs_center'
    m_mode_array = np.zeros((frame_height, frame_count), np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_count):
        _, frame = cap.read()
        colmun = np.zeros((frame_height, mean_width), np.uint8)
        for j in range(mean_width):
            # calc. average in each rows(= num of mean_width)
            colmun[:, j] = frame[:, cs_center - int(mean_width / 2) + j, 0]
        merge = np.average(colmun, axis=1)
        m_mode_array[:, i] = merge
    return m_mode_array


def set_var():
    global cs_center
    global mean_width
    cs_center = scale.get()
    mean_width = list_.get()
    if (abs(cs_center - 150) + int(mean_width) < 150):
        sub.quit()
    else:
        messagebox.showerror('Selection error...',
                             "Please retry to select cross-section region.")


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

# Set boundary conditions
sub = tkinter.Toplevel()
sub.title('Please set boundary conditions...')
sub.columnconfigure(0, weight=1)
sub.rowconfigure(0, weight=1)

sub_frame = ttk.Frame(sub, padding=10)
sub_frame.grid(sticky=(N, W, S, E))
sub_frame.columnconfigure(0, weight=1)
sub_frame.rowconfigure(0, weight=1)

# Get first 10% frame from first video file
cap = cv2.VideoCapture(files[0])
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 10))
_, frame = cap.read()
cap_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
cap_frame_pil = Image.fromarray(cap_frame_rgb)  # RGB to PIL format
echo_image = ImageTk.PhotoImage(cap_frame_pil)  # PIL to ImageTk format

label_fig = ttk.Label(
    sub_frame,
    image=echo_image
)
label_fig.grid(row=0, column=0, sticky=(N, E, S, W))

mean_val = StringVar()
mean_val.set('3')
list_ = ttk.Spinbox(
    sub_frame,
    format='%3.0f',
    state='readonly',
    textvariable=mean_val,
    from_=1,
    to=200,
    increment=1
)
list_.grid(row=0, column=1, sticky=(E))

var_scale = tkinter.IntVar()
var_scale.set(150)
scale = ttk.Scale(
    sub_frame,
    variable=var_scale,
    orient=HORIZONTAL,
    length=200,
    from_=0,
    to=299
)
scale.grid(row=1, column=0, sticky=(N, E, S, W))

button = ttk.Button(
    sub_frame,
    text='OK',
    command=lambda: set_var()
)
button.grid(row=1, column=1, padx=5, sticky=(E))

sub.mainloop()

for file in tqdm(files):
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

    cv2.imwrite(os.path.join(path, filename + '_mMode.jpg'), m_mode_array)
    cap.release()

messagebox.showinfo('Done!', "All Echo videos have been converted.")
