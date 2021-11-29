from tkinter import messagebox
import os
import time
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
inst.write(':WAV:TRACE 1')
inst.write(':WAV:STAR 0')
inst.write(':WAV:END 12499')

# get data informations
wav_start = int(inst.query(':WAV:STAR?'))
wav_end = int(inst.query(':WAV:END?'))
data_offset = float(inst.query(':WAV:OFFS?'))
data_range = float(inst.query(':WAV:RANG?'))
num_record = 1 - int(inst.query(':WAV:REC? MIN'))
print(data_offset, data_range, num_record)

# get waveform
multi_data = np.zeros((num_record + 1, 12500), float)
x = np.linspace(0, 12499, 12500)
multi_data[0, :] = x
for i in tqdm(range(num_record), leave=False):
    record = ':WAV:REC ' + str(i)
    inst.write(record)
    values = inst.query_binary_values(':WAV:SEND?', datatype='h', data_points=12500)
    multi_data[i + 1, :] = np.multiply((data_range / 3200), values) + data_offset

plt.plot(x, multi_data[1])
plt.show()

""" # save waves as csv file
path ='G:\共有ドライブ\BU301_超音波\グランドプラン\07.膀胱・筋肉量\評価チーム\oscillo_raw'
date = time.time()
filename = time.time()
full_path = os.path.join(path + date, filename + '.csv')
with open(full_path, 'w', newline='') as out_file:
    writer_output = csv.writer(out_file)
    writer_output.writerows(multi_data) """

# finish and clean
rm.close()