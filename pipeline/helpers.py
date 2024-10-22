import os.path

import torch
import logging
import librosa
from multiprocessing import Pool
import soundfile as sf
from os.path import isfile, join
from os import listdir

RANDOM_SEED = 1337
SAMPLE_RATE_AST = 32000
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 3 # seconds
SPEC_SHAPE = (298, 128) # width x height
FMIN = 20
FMAX = 15000
MAX_AUDIO_FILES = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_chunk(args):
    #Function to save a single chunk of audio data to a file.
    chunk, save_path, rate = args
    sf.write(save_path, chunk, rate)

def split_signals(filepath, output_dir, signal_length=15, n_processes=None):
    files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]

    for file in files:
        sig, rate = librosa.load(file, sr=None)
        sig_splits = [sig[i:i + int(signal_length * rate)] for i in range(0, len(sig), int(signal_length * rate)) if len(sig[i:i + int(signal_length * rate)]) == int(signal_length * rate)]

        with Pool(processes=n_processes) as pool:
            args_list = []
            for s_cnt, chunk in enumerate(sig_splits):
                save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file))[0]}_{s_cnt}.wav")
                args_list.append((chunk, save_path, rate))
            
            # Save each chunk in parallel
            pool.map(save_chunk, args_list)

