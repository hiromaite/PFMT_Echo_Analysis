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


def set_var():
    global cs_center
    global mean_width
    cs_center = scale.get()
    mean_width = list_.get()
    if abs(cs_center - 150) + int(mean_width) < 150:
        sub.withdraw()
        sub.quit()
    else:
        messagebox.showerror('Selection error...',
                             "Please retry to select cross-section region.")


def set_process():
    global flag_process
    flag_process = process.get()
    main.withdraw()
    main.quit()


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

# Ask for processing details and display the selected data information
main = tkinter.Toplevel()
main.title("Choose process...")
main.columnconfigure(0, weight=1)
main.rowconfigure(0, weight=1)

main_frame = ttk.Frame(main, padding=10)
main_frame.grid(sticky=(N, W, S, E))
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

basename = [""]
for i in range(len(files)):
    # basename.append(os.path.basename(files[i]))
    basename += os.path.basename(files[i])
    basename += "\n"
basename += "[EOF]"
basename = "".join(basename)
# print(basename)

label_info = ttk.Labelframe(
    main_frame,
    text='Loaded data(s)',
    padding=(10),
    style='My.TLabelframe')
label_info.grid(row=0, column=0, sticky=(N, S, W, E))

label_datainfo = ttk.Label(
    label_info,
    text=basename,
    padding=(5)
)
label_datainfo.grid(sticky=(N, S, W, E))

radio_frame = ttk.Frame(main_frame, padding=5)
radio_frame.grid(row=0, column=1, sticky=(N, S, W, E))

process = StringVar()
rb1 = ttk.Radiobutton(
    radio_frame,
    text="M-mode only",
    value='M',
    variable=process)
rb1.grid(row=0, column=0, sticky=(S, W))

rb2 = ttk.Radiobutton(
    radio_frame,
    text="Binarization with M-mode",
    value='B',
    variable=process)
rb2.grid(row=1, column=0, sticky=(N, S, W))

rb3 = ttk.Radiobutton(
    radio_frame,
    text="Calc. the amount of elevation",
    value='E',
    variable=process)
rb3.grid(row=2, column=0, sticky=(N, W))
process.set('B')

buttons = ttk.Frame(main_frame, padding=(0, 5))
buttons.grid(row=1, column=0, columnspan=2, sticky=(N, S))

button1 = ttk.Button(
    buttons, text='OK',
    command=lambda: set_process()
)
button1.pack(side=LEFT)

button2 = ttk.Button(buttons, text='Cancel', command=quit)
button2.pack(side=LEFT)

main.mainloop()

# Set boundary conditions
sub = tkinter.Toplevel()
sub.title("Please set boundary conditions...")
sub.columnconfigure(0, weight=1)
sub.rowconfigure(0, weight=1)

sub_frame = ttk.Frame(sub, padding=10)
sub_frame.grid(sticky=(N, W, S, E))
sub_frame.columnconfigure(0, weight=1)
sub_frame.rowconfigure(0, weight=1)

# Get first 10% frame from first video file
cap_init = cv2.VideoCapture(files[0])
cap_init.set(cv2.CAP_PROP_POS_FRAMES,
             int(cap_init.get(cv2.CAP_PROP_FRAME_COUNT) / 10))
_, frame = cap_init.read()
cap_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
cap_frame_pil = Image.fromarray(cap_frame_rgb)  # RGB to PIL format
echo_image = ImageTk.PhotoImage(cap_frame_pil)  # PIL to ImageTk format
cap_init.release()

label_fig = ttk.Label(
    sub_frame,
    image=echo_image
)
label_fig.grid(row=0, column=0, columnspan=2, sticky=(N, E, S, W))

label_msg = ttk.Label(
    sub_frame,
    text="Use the slider(above) to select the position, and select the processing width from the pull-down list(below).",
    justify='left',
    wraplength=300
)
label_msg.grid(row=2, column=0, columnspan=2, sticky=(N, E, S, W))

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
list_.grid(row=3, column=0, sticky=(E))

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
scale.grid(row=1, column=0, columnspan=2, sticky=(N, E, S, W))

button = ttk.Button(
    sub_frame,
    text='OK',
    command=lambda: set_var()
)
button.grid(row=3, column=1, padx=5, sticky=(E))

sub.mainloop()

# Execution portion
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
    if flag_process == 'M':
        cv2.imwrite(os.path.join(path, filename + '_mMode.jpg'), m_mode_array)

    # binarize from m-mode picture
    if flag_process == 'B' or flag_process == 'E':
        m_mode_th = np.zeros((frame_height, frame_count, 3), np.uint8)
        m_mode_gaussian = cv2.GaussianBlur(m_mode_array, (5, 5), 0)
        th, m_mode_th = cv2.threshold(
            m_mode_gaussian, 40, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(path, filename + '_mMode_binary_th' + str(th) + '.jpg'),
                    m_mode_th)

        m_mode_hl = m_mode_array[:, :, :]
        # print(m_mode_th[80:, 0, 0])
        # print(m_mode_th[80:, 0, 0].tolist())
        for i in range(frame_count):
            idx_bob = m_mode_th[80:, i, 0].tolist().index(255)
            m_mode_hl[idx_bob + 80, i, 1:2] = 0
            m_mode_hl[idx_bob + 80, i, 2] = 255
        cv2.imwrite(os.path.join(path, filename + '_mMode_highlight_th' + str(th) + '.jpg'),
                    m_mode_hl)

    cap.release()

#messagebox.showinfo("Done!", "All Echo videos have been converted.")
