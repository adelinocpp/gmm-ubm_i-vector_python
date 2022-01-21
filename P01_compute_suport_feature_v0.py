#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:51:50 2021

@author: adelino
"""

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from DB_wav_reader import find_wavs
import configure as c
import os
import librosa
from utils import vad
import math

Compute_Train_Features = True
Compute_Test_Features = True
win_length=0.025
hop_length=0.01
n_win_length = math.ceil(8000*win_length)
n_FFT = 2 ** math.ceil(math.log2(n_win_length))

if (Compute_Train_Features):
    file_list = find_wavs(c.TRAIN_WAV_DIR)
    file_list.sort()
    splitDir = c.TRAIN_SUPORT_FEATURES_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    
    for filename in file_list:            
        signal = basic.SignalObj(filename)
        audio, sr = librosa.load(filename, sr=None, mono=True)
        pitch = pYAAPT.yaapt(signal, **{'bp_low':30.0, 'f0_min' : 50.0,'f0_max' : 600.0, 'frame_length' : 25.0, 'frame_space' : 10.0})
        vad_sohn = vad.VAD(audio, sr, nFFT=n_FFT, win_length=win_length, \
                       hop_length=hop_length, theshold=0.7)