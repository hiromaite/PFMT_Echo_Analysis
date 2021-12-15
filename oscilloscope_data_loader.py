from tkinter import messagebox
import os
import datetime
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa
from pyvisa.constants import *
from tqdm import tqdm

# get resource list and open USB instrument
rm = visa.ResourceManager()
list_inst = rm.list_resources()
if len(list_inst) == 0:
    messagebox.showerror('Error!',
                         "Can't find instrument(s), Please check USB conection")
    exit()
else:
    flg_found = False
    for i in range(len(list_inst)):
        if list_inst[i].startswith("USB"):
            inst = rm.open_resource(list_inst[i])
            idn = inst.query('*IDN?')
            if idn.startswith("YOKOGAWA"):
                flg_found = True
                break

# check opened instrument is YOKOGAWA or not
if not flg_found:
    messagebox.showerror('Error!',
                         "YOKOGAWA oscillo is not found.")
    exit()

# set oscilloscope settings
inst.write(':COMM:HEAD 0;:WAV:FORM WORD;BYT LSBF')

# get data informations
wav_start = int(inst.query(':WAV:STAR?'))
wav_end = int(inst.query(':WAV:END?'))
wav_leng = int(inst.query(':WAV:LENG?'))
rec_len = int(inst.query(':ACQ:RLEN?'))
sample_rate = float(inst.query(':WAV:SRAT?'))
pos_trigger = int(inst.query(':WAV:TRIG?'))
time_start = (wav_start - pos_trigger) / sample_rate
time_end = (wav_end - pos_trigger) / sample_rate
num_record = 1 - int(inst.query(':WAV:REC? MIN'))
list_ch = ['1', '2', 'MATH2']

# acquire waveforms
multi_data = np.zeros((len(list_ch), num_record + 1, wav_leng), float)
x = np.linspace(time_start, time_end, wav_leng)
path = r'G:\共有ドライブ\BU301_超音波\グランドプラン\07.膀胱・筋肉量\評価チーム\oscillo_raw'
d_today = datetime.date.today()
dt_now = datetime.datetime.now()
date = str(d_today)
time = 'rawdata_{:02}{:02}{:02}'.format(
    dt_now.hour, dt_now.minute, dt_now.second)
os.makedirs(os.path.join(path, date, time), exist_ok=True)
for ch in tqdm(range(len(list_ch)), leave=False, desc="Acquiring waveforms..."):
    multi_data[ch, 0, :] = x
    inst.write(':WAV:TRACE ' + list_ch[ch])
    data_offset = float(inst.query(':WAV:OFFS?'))
    data_range = float(inst.query(':WAV:RANG?'))
    for rec in tqdm(range(num_record), leave=False):
        inst.write(':WAV:REC ' + str(1 + rec - num_record))
        values = inst.query_binary_values(
            ':WAV:SEND?', datatype='h', data_points=wav_leng)
        multi_data[ch, rec + 1, :] = np.multiply(
            (data_range / 3200), values) + data_offset

# draw movie file
path_video = os.path.join(path, date, time, 'video_out.mp4')
four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(path_video, four_cc, 8.0, (640, 480))
for frame in tqdm(range(num_record), leave=False, desc="Drawing graph plot..."):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # default dpi = 100
    ax.plot(x[::10], multi_data[1, frame, ::10])  # data points was reduced to improve drawing speed and reduce memory usage
    ax.set_xlabel("ToF [sec]")
    ax.set_ylabel("Voltage [V]")
    ax.text(0.01, 0.99, "Record num = " + str(frame),
            verticalalignment='top', transform=ax.transAxes)
    #ax.set_ylim(0, 180)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    video.write(img)
    plt.close()
video.release

# save waves as csv file
for ch in tqdm(range(len(list_ch)), leave=False, desc="Saving CSV files..."):
    full_path = os.path.join(path, date, time, 'ch.' + list_ch[ch] + '.csv')
    with open(full_path, 'w', newline='') as out_file:
        writer_output = csv.writer(out_file)
        writer_output.writerows(multi_data[ch, :, :].T)

# finish and clean
rm.close()
