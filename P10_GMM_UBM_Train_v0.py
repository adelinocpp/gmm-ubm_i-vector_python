#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:50:07 2021

@author: adelino
"""
import configure as c
from DB_wav_reader import find_feats, find_gmms
import os
import pickle
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# --- Matriz de calibracao -----------------------------------------------------
MTXFileName = os.path.join(c.GMM_UBM_SAVE_MODELS_DIR, c.CALIBRATE_MTX_FILE)
TRAIN_THR_FileName = os.path.join(c.GMM_UBM_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
# ------------------------------------------------------------------------------
ComputeNewModel = False
show_result = True

feature_list = find_feats(c.TRAIN_FEAT_DIR)
feature_list.sort()
gmm_list = find_gmms(c.GMM_TRAIN_DIR)
gmm_list.sort()
numFeature = len(feature_list)
numGMM = len(gmm_list)
# ------------------------------------------------------------------------------
if os.path.exists(MTXFileName):
    with open(MTXFileName,'rb') as fp:
        mtxScore = pickle.load(fp)
        y_deff = pickle.load(fp)
        y_pred = pickle.load(fp)
        fp.close()
else:
    print("Carregando modelos...")
    gmmData = list()
    for gmmFile in gmm_list:
        with open(gmmFile, 'rb') as f:
            gmm_and_label = pickle.load(f)
        gmmData.append(gmm_and_label)
        
    print("Calibrando GMM-UBM")
    with open(c.GMM_UBM_FILE_NAME, 'rb') as f:
        UBM = pickle.load(f)
    mtxScore = np.zeros((numFeature,numGMM))
    y_deff = np.zeros((numGMM*numFeature))
    y_pred = np.zeros((numGMM*numFeature))
    for i, featureFile in enumerate(feature_list):
        with open(featureFile, 'rb') as f:
            feat_and_label = pickle.load(f)
        labelFeat = feat_and_label['label']
        vX = feat_and_label['feat']
        predUBM = UBM.score_samples(vX)
        for j, gmm_and_label in enumerate(gmmData):
            # with open(gmmFile, 'rb') as f:
            #     gmm_and_label = pickle.load(f)
            labelGMM = gmm_and_label['label']
            GMM = gmm_and_label['model']
            predGMM = GMM.score_samples(vX)
            mtxScore[i,j] = np.mean(predGMM - predUBM)
            y_deff[i*numFeature + j] = int(labelFeat == labelGMM)
            y_pred[i*numFeature + j] = mtxScore[i,j]
        print("Calculado score de {:04d}/{:04d}".format(i,numFeature-1))
    with open(MTXFileName,'wb+') as fp:
        pickle.dump(mtxScore, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_deff, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
# ------------------------------------------------------------------------------
if (show_result):
    figureFile = os.path.join(c.GMM_UBM_SAVE_MODELS_DIR, 'TRAIN_UBM_GMM_plot.png')
    print("Gerando resultado treinamento...")
    idxSS = np.nonzero(y_deff == 1)
    idxDS = np.nonzero(y_deff == 0)
    SSscore = y_pred[idxSS]
    DSscore = y_pred[idxDS]    
    densSS = gaussian_kde(SSscore)
    densDS = gaussian_kde(DSscore)
    xss_vals = np.linspace(np.min(SSscore),np.max(SSscore),2000) 
    xds_vals = np.linspace(np.min(DSscore),np.max(DSscore),2000) 
    pdfSS = densSS.pdf(xss_vals)
    pdfDS = densDS.pdf(xds_vals)   
    fig = plt.figure(figsize=(10,8))
    plt.plot(xss_vals,pdfSS)
    plt.plot(xds_vals,pdfDS)
    plt.xlabel('score')
    plt.ylabel('prob.')
    plt.xlim((np.mean(DSscore) - 3*np.std(DSscore)), (np.mean(SSscore) + 3*np.std(SSscore))) # consistent scale
    plt.grid(True)
    fig.savefig(figureFile, bbox_inches='tight')    
# ------------------------------------------------------------------------------
print("Calculando limiares...")
fpr, tpr, threshold = roc_curve(y_deff, y_pred, pos_label=1)
fnr = 1 - tpr
idx = np.nanargmin(np.absolute((fnr - fpr)))
eer_threshold = threshold[idx]
print("fpr: {:5.3f}, tpr: {:5.3f}, fnr: {:5.3f}, thr: {:5.3f}".format(fpr[idx],tpr[idx],fnr[idx],eer_threshold ))
center_threshold = 0.5*(np.max(DSscore) + np.min(SSscore))
pond_threshold = (len(idxSS[0])*np.max(DSscore) + len(idxDS[0])*np.min(SSscore))/len(y_deff)
print("Scores - SS min: {:5.3f} DS max: {:5.3f}".format(np.min(SSscore),np.max(DSscore)))
print("Threshoold - eer: {:5.3f} center: {:5.3f} ponder: {:5.3f}".format(eer_threshold,center_threshold,pond_threshold ))
if not os.path.exists(TRAIN_THR_FileName):
    with open(TRAIN_THR_FileName,'wb') as fp:
        pickle.dump(eer_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(center_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pond_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()