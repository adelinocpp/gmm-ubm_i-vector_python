"""
Created on Fri Sep  3 14:36:59 2021

@author: adelino
"""
import configure as c
from DB_wav_reader import find_feats
# import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from utils.SpheringSVD import SpheringSVD as Sphering
# from utils.PLDA import plda as PLDA
import utils.plda as plda
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
import re
import math

# ------------------------------------------------------------------------------
def trap_int(x,y):
    npts = len(y)
    z = np.zeros((npts,))
    z[1:-1] = z[1:-1] + y[0:-2]
    z[1:-1] = z[1:-1] + y[1:-1]
    z = 0.5*(x[1] - x[0])*z
    return np.cumsum(z)
# ------------------------------------------------------------------------------
TRAIN_THR_FileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
# --- OBSERVACAO ---------------------------------------------------------------
MTXFileName = os.path.join(c.TEST_RESULTS_DIR, c.TEST_CONF_MTX)
ComputeNewModel = True
if (ComputeNewModel):
    os.remove(MTXFileName)
    
   
useSphering = True
# use_cuda = False
# embedding_size = 512
embedding_size = 128
show_result = False

Compute_Test_PLDA = True
LDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.LDA_FILE)
if (not os.path.exists(LDAFileName)):
    print("Modelo LDA não existe. Executar a rotina P06.")
    Compute_Test_PLDA = False      

SpheringFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.SPHERING_FILE)
if ((not os.path.exists(LDAFileName)) and (useSphering)):
    print("Modelo de normalização não existe. Executar a rotina P06 com useSphering = True.")
    Compute_Test_PLDA = False      

PLDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.PLDA_FILE)
if (not os.path.exists(PLDAFileName)):
    print("Modelo de PLDA não existe. Executar a rotina P06.")
    Compute_Test_PLDA = False   

TRAIN_THR_FileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
if (not os.path.exists(TRAIN_THR_FileName)):
    print("Arquivo de calibração não existe. Executar a rotina P06.")
    Compute_Test_PLDA = False   


# --- Matriz de resultados ------------------------------------------------
if (Compute_Test_PLDA):
    if (useSphering):
        with open(SpheringFileName,'rb') as fp:
            SpheModel = pickle.load(fp)
            fp.close()
            
    with open(LDAFileName,'rb') as fp:
        LDAModel = pickle.load(fp)
        fp.close()
        
    with open(PLDAFileName,'rb') as fp:
        PLDAModel = pickle.load(fp)
        fp.close()
    
    with open(TRAIN_THR_FileName,'rb') as fp:
        eer_threshold = pickle.load(fp)
        center_threshold = pickle.load(fp)
        pond_threshold = pickle.load(fp)
        fp.close()

    file_list = find_feats(c.IVECTOR_TEST_DIR)
    file_list.sort()
    embeddings = {}
    numEnroll = len(file_list)
    X = np.zeros((numEnroll,embedding_size))
    y = np.zeros((numEnroll,))
    for i, file in enumerate(file_list):
        spk = int(file.split('/')[-1].split('_')[1]) # filename: DIR_OF_FILE/L_####_U_####_.p
        y[i] = spk;
        # print("i,spk ({:d},{:d})".format(i,spk))
        with open(file, "rb") as fp: 
            embeddings[spk] = pickle.load(fp)
            fp.close()
            # embeddings[spk] = torch.load(file)
            if ((i == 0) and (not embeddings[spk].shape[0] == embedding_size)):
                embedding_size = embeddings[spk].shape[0]
                X = np.zeros((numEnroll,embedding_size))
    
            X[i,:] = np.array(embeddings[spk])
    print('Carregados {:03d} locutores com {:04d} i-vectors'.format(len(embeddings),len(y)))

    if (not os.path.exists(MTXFileName)):
        # --- processo de Sphearing --------------------------------------------
        if (useSphering):
            Xsph = SpheModel.transform(X)
        else:
            Xsph = X
        # --- processo LDA -----------------------------------------------------
        Xlda = LDAModel.transform(Xsph)
    
        U_model = PLDAModel.model.transform(Xlda, from_space='D', to_space='U_model')
        hEnroll = math.ceil(0.5*numEnroll)
        
        mtxScore = np.zeros((hEnroll,hEnroll))
        y_deff = np.zeros((hEnroll*hEnroll))
        y_pred = np.zeros((hEnroll*hEnroll))
        iK = 0;
        k = 0;
        for i in range(0,numEnroll,2):
            U_datum_0 = U_model[i][None,]
            jK = 0;
            for j in range(1,numEnroll,2):
                U_datum_1 = U_model[j][None,]
                log_ratio_0_1 = PLDAModel.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
                mtxScore[iK,jK] = log_ratio_0_1
                print("Scorring ({:03d},{:03d}) ({:06d})".format(i,j,i*numEnroll + j))
                y_deff[k] = int(y[i] == y[j])
                y_pred[k] = mtxScore[iK,jK] 
                jK += 1
                k += 1
            iK += 1
        idxSS = np.nonzero(y_deff == 1)
        idxDS = np.nonzero(y_deff == 0)
        SSscore = y_pred[idxSS]
        DSscore = y_pred[idxDS]
        
        splitDir = c.TEST_RESULTS_DIR.split('/')
        for idx in range(1,len(splitDir)+1):
            curDir = '/'.join(splitDir[:idx])
            if (not os.path.exists(curDir)):
                os.mkdir(curDir)
        
        with open(MTXFileName,'wb') as fp:
            pickle.dump(mtxScore, fp, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_deff, fp, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
    else:
        with open(MTXFileName,'rb') as fp:
            mtxScore = pickle.load(fp)
            y_deff = pickle.load(fp)
            y_pred = pickle.load(fp)
            fp.close()
        idxSS = np.nonzero(y_deff == 1)
        idxDS = np.nonzero(y_deff == 0)
        SSscore = y_pred[idxSS]
        DSscore = y_pred[idxDS]
        
    if (show_result):  
        print("Gerando gráfico do resultado...")         	
        densSS = gaussian_kde(SSscore)
        densDS = gaussian_kde(DSscore)
        xss_vals = np.linspace(np.min(y_pred),np.max(y_pred) + np.std(y_pred),10000) 
        xds_vals = np.linspace(np.min(y_pred),np.max(y_pred) + np.std(y_pred),10000) 
        pdfSS = densSS.pdf(xss_vals)
        pdfDS = densDS.pdf(xds_vals)
    
        cdfSS = trap_int(xss_vals,pdfSS)
        cdfDS = trap_int(xds_vals,pdfDS)
    
        fig = plt.figure(figsize=(10,8))
        plt.plot(xss_vals,cdfSS)
        plt.plot(xds_vals,1-cdfDS)
        plt.xlabel('score')
        plt.ylabel('acum prob.')
    #    plt.ylim(0, max(xss_vals)) # consistent scale
        plt.xlim((np.mean(DSscore) - 1.96*np.std(DSscore)), (np.mean(SSscore) + 1.96*np.std(SSscore))) # consistent scale
        plt.grid(True)
        # plt.show()
        fig.savefig('TEST_plot_iVector_LDA_PLDA_00.png', bbox_inches='tight')
    
        fig = plt.figure(figsize=(10,8))
        plt.plot(xss_vals,pdfSS)
        plt.plot(xds_vals,pdfDS)
        plt.xlabel('score')
        plt.ylabel('prob.')
    #    plt.ylim(0, max(SSscore)) # consistent scale
        plt.xlim((np.mean(DSscore) - 1.96*np.std(DSscore)), (np.mean(SSscore) + 1.96*np.std(SSscore))) # consistent scale
        plt.grid(True)
        # plt.show()
        fig.savefig('TEST_plot_iVector_LDA_PLDA_01.png', bbox_inches='tight')    

    with open(TRAIN_THR_FileName,'rb') as fp:
        eer_threshold = pickle.load(fp)
        center_threshold = pickle.load(fp)
        pond_threshold = pickle.load(fp)
        fp.close()    
    
    P = len(idxSS[0])
    N = len(idxDS[0])
    idxTP = np.nonzero(SSscore > eer_threshold)
    idxTN = np.nonzero(DSscore <= eer_threshold)
    TP = len(idxTP[0])
    FN = P - TP    
    TN = len(idxTN[0])
    FP = N - TN
    TPR = TP/P
    TNR = TN/N
    FPR = 1-TNR
    FNR = 1-TPR
    ACC = (TP+TN)/(P+N)
    print("TAXAS eer_threshold: {:5.3f}".format(eer_threshold))
    print("TPR: {:5.3f}, TNR: {:5.3f}, FPR: {:5.3f}, FNR: {:5.3f}, ACC: {:5.3f}".format(TPR,TNR,FPR,FNR,ACC))
    
    idxTP = np.nonzero(SSscore > center_threshold)
    idxTN = np.nonzero(DSscore <= center_threshold)
    TP = len(idxTP[0])
    FN = P - TP    
    TN = len(idxTN[0])
    FP = N - TN
    TPR = TP/P
    TNR = TN/N
    FPR = 1-TNR
    FNR = 1-TPR
    ACC = (TP+TN)/(P+N)
    print("TAXAS center_threshold: {:5.3f}".format(center_threshold))
    print("TPR: {:5.3f}, TNR: {:5.3f}, FPR: {:5.3f}, FNR: {:5.3f}, ACC: {:5.3f}".format(TPR,TNR,FPR,FNR,ACC))
    
    idxTP = np.nonzero(SSscore > pond_threshold)
    idxTN = np.nonzero(DSscore <= pond_threshold)
    TP = len(idxTP[0])
    FN = P - TP    
    TN = len(idxTN[0])
    FP = N - TN
    TPR = TP/P
    TNR = TN/N
    FPR = 1-TNR
    FNR = 1-TPR
    ACC = (TP+TN)/(P+N)
    print("TAXAS pond_threshold: {:5.3f}".format(pond_threshold))
    print("TPR: {:5.3f}, TNR: {:5.3f}, FPR: {:5.3f}, FNR: {:5.3f}, ACC: {:5.3f}".format(TPR,TNR,FPR,FNR,ACC))
