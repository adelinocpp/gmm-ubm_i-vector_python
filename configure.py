# Caminho para os arquivos de audio
TRAIN_WAV_DIR = '../../00_Bases_de_Dados/CEFALA_8k_Refinados'
TEST_WAV_DIR = '../../00_Bases_de_Dados/WhatsApp_01_8k_wav'

# Caminho para as caracteristicas calculadas
TRAIN_FEAT_DIR = 'feat_logfbank_nfilt40/train'
TEST_FEAT_DIR = 'feat_logfbank_nfilt40/test'

# Caminho para os dados da analise LDA/PLDA
LDA_SAVE_MODELS_DIR = 'LDA_saved/'
LDA_FILE = 'lda_model.txt'
SPHERING_FILE = 'sphering_model.txt'
PLDA_FILE = 'plda_model.txt'
CALIBRATE_MTX_FILE = 'afinity_matrix.txt'
CALIBRATE_THR_FILE = 'threshoold.txt'


TEST_RESULTS_DIR = 'test_results/'
TEST_CONF_MTX = 'conf_mtx_test.txt'


UBM_FILE_NAME = 'ubm_data_file.p'
SAVE_MODELS_DIR = 'model_saved'
# Context window size
NUM_WIN_SIZE = 250 #10 # 100

# LNORM
ALPHA_LNORM = 10
# Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = True
USE_SCALE = False
SAMPLE_RATE = 8000
FILTER_BANK = 40

# Settings for GMM-UBM
nComponents = 512
covType='diag'
GMM_UBM_FILE_NAME='gmm_ubm_model_file.p'
GMM_TRAIN_DIR = 'gmm_models/train'
GMM_TEST_DIR = 'gmm_models/train'
GMM_UBM_SAVE_MODELS_DIR = 'gmm_models/'

TEST_RESULTS_DIR = 'test_results/'
TEST_CONF_MTX = 'conf_mtx_test.txt'

IVECTOR_TRAIN_DIR = 'ivector_models/train'
IVECTOR_TEST_DIR = 'ivector_models/test'
T_MATRIX_FILE_NAME = 'T_matrix_file.p'

BW_SCALER_FILE = 'BW_Scaler_for_T_matrix.p'

TRAIN_SUPORT_FEATURES_DIR = 'suport_feature/train'
TEST_SUPORT_FEATURES_DIR = 'suport_feature/test'
