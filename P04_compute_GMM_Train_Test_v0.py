#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:58:00 2021

@author: adelino
"""
DEBUG_MODE = False
if (DEBUG_MODE):
    import sys
import configure as c
import pickle
from DB_wav_reader import find_feats
import os
from sklearn.mixture import GaussianMixture
import numpy as np

# -----------------------------------------------------------------------------
def mapAdapt(vX, UBM):
    posterioriX = UBM.predict_proba(vX)    
    mapAdaptRelevance = 19
    vec_sumPosP = posterioriX.sum(0)
    vec_sumPosMu = np.matmul(posterioriX.T,vX)
    vec_sumPosDSig = np.matmul(posterioriX.T,np.power(vX,2))
    sum_N = vec_sumPosP.sum(0)
    sum_W = 0
    m_Rho = np.zeros(UBM.n_components)
    m_Mu = np.zeros([UBM.n_components, vX.shape[1]])
    m_Sig = np.zeros([UBM.n_components, vX.shape[1]])
    for k in range(0,UBM.n_components):
        dbl_alpha = vec_sumPosP[k]/(vec_sumPosP[k] + mapAdaptRelevance)
        dbl_Wtemp = UBM.weights_[k]*(1-dbl_alpha)  + dbl_alpha*vec_sumPosP[k]/sum_N
        sum_W += dbl_Wtemp
        m_Rho[k] = dbl_Wtemp
        m_Mu[k,:] = UBM.means_[k,:]*(1-dbl_alpha) + dbl_alpha*vec_sumPosMu[k,:]/(vec_sumPosP[k] + 1e-12)
        m_Sig[k,:] = (UBM.covariances_[k,:] + np.power(UBM.means_[k,:],2))*(1-dbl_alpha) + \
                        dbl_alpha*vec_sumPosDSig[k,:]/(vec_sumPosP[k] + 1e-12) - np.power(m_Mu[k,:],2)
    
    m_Rho = m_Rho/(m_Rho.sum(0))
    GMM = GaussianMixture(n_components = UBM.n_components, covariance_type=UBM.covariance_type)
    GMM.weights_ = m_Rho
    GMM.means_ = m_Mu
    GMM.covariances_ = m_Sig
    GMM.precisions_ = 1/m_Sig
    GMM.precisions_cholesky_ = 1/m_Sig
    return GMM
# -----------------------------------------------------------------------------        
Compute_GMM_Train = False
Compute_GMM_Test = True

if (not os.path.exists(c.TRAIN_FEAT_DIR)):
    print("Diretório de características de treinamento não existe. Executar a rotina P01.")
    Compute_GMM_Train = False

if (not os.path.exists(c.TEST_FEAT_DIR)):
    print("Diretório de características de teste não existe. Executar a rotina P01.")
    Compute_GMM_Train = False
    
if (not os.path.exists(c.GMM_UBM_FILE_NAME)):
    print("Arquivo UBM não existe. Executar a rotina P02.")
    Compute_GMM_Train = False
    Compute_GMM_Test = True
    
if (Compute_GMM_Train):
    with open(c.GMM_UBM_FILE_NAME, 'rb') as f:
        UBM = pickle.load(f)
        
    file_list = find_feats(c.TRAIN_FEAT_DIR)
    file_list.sort()
    splitDir = c.GMM_TRAIN_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
            
    print('Inicio do calculo dos GMM:')
    print('Arquivos de treinamento: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
            
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.GMM_TRAIN_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        gmm_and_label = {}
        gmm_and_label['label'] = feat_and_label['label']
        gmm_and_label['model'] = mapAdapt(feat_and_label['feat'],UBM)
        if (not os.path.exists(c.GMM_TRAIN_DIR + '/' + filenameFolder)):
            os.mkdir(c.GMM_TRAIN_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(gmm_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        if (DEBUG_MODE):
            sys.exit("MODO DEPURACAO: Fim do script")
    
if (Compute_GMM_Test):
    with open(c.GMM_UBM_FILE_NAME, 'rb') as f:
        UBM = pickle.load(f)
    
    file_list = find_feats(c.TEST_FEAT_DIR)
    file_list.sort()
    splitDir = c.GMM_TEST_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
            
    print('Inicio do calculo dos GMM:')
    print('Arquivos de teste: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
            
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.GMM_TEST_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        gmm_and_label = {}
        gmm_and_label['label'] = feat_and_label['label']
        gmm_and_label['model'] = mapAdapt(feat_and_label['feat'],UBM)
        if (not os.path.exists(c.GMM_TEST_DIR + '/' + filenameFolder)):
            os.mkdir(c.GMM_TEST_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(gmm_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        if (DEBUG_MODE):
            sys.exit("MODO DEPURACAO: Fim do script")
    
