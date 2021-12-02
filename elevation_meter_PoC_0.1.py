import tkinter
from tkinter import filedialog, messagebox
import os

import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# time scale
x = np.linspace(0, 12499, 12500)
x = x - 1250  # trigger position
x = x * (20 * 0.001 * 0.001) / 1250  # sampling rate
x = x * 1520 * 1000  # speed of sound

# attenuation coefficient
db = (x / 10) * -4
attn = 10 ** (db / 20)



#test output
attnd = multi_data[0] / attn
fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
ax.plot(x, attnd)
#ax.plot(x, multi_data[0])
# ax.set_ylim(0, 180)
ax.set_xlabel("Depth [mm]")
ax.set_ylabel("Intensity [arb. Unit]")
fig.show()
messagebox.showinfo('Information', "Execution has done")