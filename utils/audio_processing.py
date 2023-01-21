import os

import librosa
import numpy as np
import skimage
import torch
import torchaudio
import torchaudio.transforms as T

import noisereduce as nr

from consts import SAMPLE_RATE, N_FFT, FRAME_SIZE, HOP_SIZE, N_MELS, WINDOW_FN, POWER, PAD_MODE, NORM, MEL_SCALE, \
    THRESH_N_NR, STATIONARY_NR
from utils import empty_folder, scale_minmax

mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=FRAME_SIZE,
    hop_length=HOP_SIZE,
    n_mels=N_MELS,
    window_fn=WINDOW_FN,
    power=POWER,
    pad_mode=PAD_MODE,
    norm=NORM,
    mel_scale=MEL_SCALE,
)


def audio_sample_cutout(audio_path, start_second, end_second):
    y, sr = torchaudio.load(audio_path)

    start = start_second * sr
    end = end_second * sr
    waveform = y[:, start:end]

    return waveform, sr


def process_audio_sample(audio_path, data_path):
    empty_folder(os.path.join(data_path, 'test'))
    audio_sample_to_spectrograms(audio_path, os.path.join(data_path, 'test'))


def waveform_to_spectrogram(waveform, sample_rate):
    preprocessed = nr.reduce_noise(y=waveform, sr=sample_rate,
                                   thresh_n_mult_nonstationary=THRESH_N_NR,
                                   stationary=STATIONARY_NR)
    mel_spec = mel_spectrogram(torch.from_numpy(preprocessed))

    return librosa.power_to_db(mel_spec).transpose((1, 2, 0))


def single_sample_to_spectrogram(waveform, sample_rate, destination_path, filename, preprocessing=True):
    if preprocessing:
        temp = nr.reduce_noise(y=waveform, sr=sample_rate,
                               thresh_n_mult_nonstationary=THRESH_N_NR,
                               stationary=STATIONARY_NR)
        waveform = torch.from_numpy(temp)

    mel_spec = mel_spectrogram(waveform)
    db_spec = librosa.power_to_db(mel_spec).transpose((1, 2, 0))
    scaled = scale_minmax(db_spec, 0, 255).astype(np.uint8)

    skimage.io.imsave(os.path.join(destination_path, f'{filename}.png'), scaled)


def audio_sample_to_spectrograms(audio_path, destination_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    for i in range(waveform.shape[1] * 2 // sample_rate - 1):
        start = i * sample_rate // 2
        end = i * sample_rate // 2 + sample_rate
        y = waveform[:, start:end]

        single_sample_to_spectrogram(y, sample_rate, destination_path, f'{i / 2}s')


def audio_dataset_to_spectrogram_dataset(source_path, dest_path):
    dataset_splits = ['train', 'val', 'test']
    labels = [label for label in os.listdir(os.path.join(source_path, 'train')) if not label.startswith('.')]

    for split in dataset_splits:
        for label in labels:
            audio_source_path = os.path.join(source_path, split, label)
            files = [file for file in os.listdir(audio_source_path) if not file.startswith('.')]

            for filename in files:
                waveform, sample_rate = torchaudio.load(os.path.join(audio_source_path, filename))

                new_filename = filename.split('.')[0] + '_' + filename.split('.')[1]
                single_sample_to_spectrogram(waveform, sample_rate, os.path.join(dest_path, split, label), new_filename)
