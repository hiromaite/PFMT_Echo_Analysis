from tkinter import messagebox
import os
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
inst.write(':WAVEFORM:TRACE 2')

#get data informations
data_offset = inst.query(':WAV:OFFS?')
data_range = inst.query(':WAV:RANG?')
num_record = 1 - str(inst.query(':WAVeform:RECord? MINimum'))