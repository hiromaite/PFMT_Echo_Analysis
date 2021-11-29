from tkinter import messagebox
import os
import datetime
import csv

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
    for i in range(len(list_inst)):
        if list_inst[i].startswith("USB"):
            inst = rm.open_resource(list_inst[i])
            break

# check opened instrument is YOKOGAWA or not
idn = inst.query('*IDN?')
if not idn.startswith("YOKOGAWA"):
    messagebox.showerror('Error!',
                         "Opened instrument is not YOKOGAWA oscillo.")
    exit()

# set oscilloscope settings
inst.write(':COMM:HEAD 0;:WAV:FORM WORD;BYT LSBF')

# get data informations
wav_start = int(inst.query(':WAV:STAR?'))
wav_end = int(inst.query(':WAV:END?'))
wav_leng = int(inst.query(':WAV:LENG?'))
rec_len = int(inst.query(':ACQ:RLEN?'))
sample_rate = float(inst.query(':WAV:SRAT?'))
data_offset = float(inst.query(':WAV:OFFS?'))
data_range = float(inst.query(':WAV:RANG?'))
num_record = 1 - int(inst.query(':WAV:REC? MIN'))
list_ch = [1, 2, 'MATH2']
print(data_offset, data_range, num_record)

# get waveform
multi_data = np.zeros((len(list_ch), num_record + 1, wav_leng), float)
x = np.linspace(0, 12499, 12500)
path ='G:\共有ドライブ\BU301_超音波\グランドプラン\07.膀胱・筋肉量\評価チーム\oscillo_raw'
d_today = datetime.date.today()
dt_now = datetime.datetime.now()
date = str(d_today)
time = 'rawdata_{:02}{:02}{:02}'.format(dt_now.hour, dt_now.minute, dt_now.second)
os.makedirs(os.path.join(path, date, time), exist_ok=True)
for ch in tqdm(range(len(list_ch)), leave=False, desc="Acquiring waveforms..."):
    multi_data[ch, 0, :] = x
    inst.write(':WAV:TRACE ' + str(list_ch[ch]))
    for rec in tqdm(range(num_record), leave=False):
        record = ':WAV:REC ' + str(rec)
        inst.write(record)
        values = inst.query_binary_values(':WAV:SEND?', datatype='h', data_points=wav_leng)
        multi_data[ch, rec + 1, :] = np.multiply((data_range / 3200), values) + data_offset

# save waves as csv file
for ch in tqdm(range(len(list_ch)), leave=False, desc="Saving CSV files..."):
    full_path = os.path.join(path, date, time, 'ch.' + str(list_ch[ch]) + '.csv')
    with open(full_path, 'w', newline='') as out_file:
        writer_output = csv.writer(out_file)
        writer_output.writerows(multi_data[ch, :, :])

# finish and clean
rm.close()