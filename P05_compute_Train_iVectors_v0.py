#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:12:21 2021

@author: adelino
"""
DEBUG_MODE = False
if (DEBUG_MODE):
    import sys
import configure as c
import pickle
import os
from DB_wav_reader import find_feats
import numpy as np
from utils.compute_bw_stats import compute_bw_stats
from utils.extract_ivector import extract_ivector
# -----------------------------------------------------------------------------        
Compute_BW_stats_T_matrix = True
if (not os.path.exists(c.TRAIN_FEAT_DIR)):
    print("Diretório de características não existe. Executar a rotina P01.")
    Compute_BW_stats_T_matrix = False        
    
if (not os.path.exists(c.GMM_UBM_FILE_NAME)):
    print("Arquivo UBM não ewxiste. Executar a rotina P02.")
    Compute_BW_stats_T_matrix = False


if (not os.path.exists(c.T_MATRIX_FILE_NAME)):
    print("Arquivo da matriz de variabildiade total (matrix T) não existe. Executar a rotina P04.")
    Compute_BW_stats_T_matrix = False
   
if (Compute_BW_stats_T_matrix):
    with open(c.GMM_UBM_FILE_NAME, 'rb') as f:
        UBM = pickle.load(f)
    with open(c.T_MATRIX_FILE_NAME, 'rb') as f:
        T = pickle.load(f)
        
    file_list = find_feats(c.TRAIN_FEAT_DIR)
    file_list.sort()
    
    splitDir = c.IVECTOR_TRAIN_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    
    # nStop = 3
    # statsTrain = np.zeros((nStop,UBM.n_components*(UBM.n_features_in_ + 1) ) )
    statsTrain = np.zeros((len(file_list),UBM.n_components*(UBM.n_features_in_ + 1) ) )
    uttCount = 0
    currLoc = 0
    lastDir = 'dir'
    for idx, filename in enumerate(file_list):
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        
        if (not (filenameFolder == lastDir)):
            currLoc += 1
            lastDir = filenameFolder
            uttCount = 1
        else:
            uttCount += 1
            
        locFileData = "L_{:04d}_U_{:04d}.p".format(currLoc,uttCount)
        filenameSave = c.IVECTOR_TRAIN_DIR + '/' + filenameFolder + '/' + locFileData
        N, F = compute_bw_stats(feat_and_label['feat'], UBM);
        x = extract_ivector(np.append(N,F),UBM,T)
        if (not os.path.exists(c.IVECTOR_TRAIN_DIR + '/' + filenameFolder)):
            os.mkdir(c.IVECTOR_TRAIN_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(x,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        
        