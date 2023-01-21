import torch

CLASS_NAMES = ['no_speech', 's1_EK', 's2_Konrad', 's3_Kaja', 's4_Radek', 'unknown']

SAMPLE_RATE = 48000
N_FFT = 2048
FRAME_SIZE = 1024
HOP_SIZE = 215
N_MELS = 224
WINDOW_FN = torch.hamming_window
POWER = 2
PAD_MODE = 'reflect'
NORM = 'slaney'
MEL_SCALE = 'htk'
RESAMPLING_METHOD = 'kaiser_best'

STATIONARY_NR = False
THRESH_N_NR = 2

DATA_ROOT = '../data/data'
TEST_DATA_ROOT = '../data/spectrograms_to_analyze'
INPUT_FILE = "../audio/test_sample.wav"
MODEL_PATH = '../models/m_squeezenet_epochs101.pt'

BATCH_SIZE = 8
EPOCHS = 101
CLASSES = len(CLASS_NAMES)

THRESHOLD = 70
