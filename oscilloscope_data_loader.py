import numpy as np
from tkinter import messagebox
import os
import time
import csv

import pyvisa as visa
from pyvisa.constants import *
import tqdm

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

# if the buffer size is not enough, specify it here
# inst.set_buffer(inst, VI_READ_BUF, 512000)
# inst.set_buffer(inst, VI_WRITE_BUF, 512000)

# set oscilloscope settings
inst.write(':COMM:HEAD 0;:WAV:FORM WORD;BYT LSBF')
inst.write(':WAVE:TRACE 2')
inst.write(':WAVE:START 0')
inst.write(':WAVE:END 12499')

# get data informations
data_offset = inst.query(':WAV:OFFS?')
data_range = inst.query(':WAV:RANG?')
num_record = 1 - str(inst.query(':WAVeform:RECord? MINimum'))

# set query settings
inst.values_format.is_binary = True
inst.values_format.datatype = 'i'
inst.values_format.is_big_endian = True
inst.values_format.container = np.array

# get waveform
multi_data = np.zeros(num_record, 12500)
for i in tqdm(range(num_record)):
    record = ':WAVEFORM:RECORD ' + str(i)
    inst.write(record)
    values = inst.query_values(':WAV:SEND?')
    multi_data[i, :] = (data_range * values / 3200) + data_offset

# save waves as csv file
path ='G:\共有ドライブ\BU301_超音波\グランドプラン\07.膀胱・筋肉量\評価チーム\oscillo_raw'
date = time.time()
filename = time.time()
full_path = os.path.join(path + date, filename + '.csv')
with open(full_path, 'w', newline='') as out_file:
    writer_output = csv.writer(out_file)
    writer_output.writerows(multi_data)

# finish and clean
rm.close()