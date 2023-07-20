"""
This is the utils module.

This module contains functions that help downloading an uploading the data from and to s3 buckets, as well as creating a bucket.
"""

# import boto3
# import logging
# import os
# import tarfile
# from botocore.exceptions import ClientError
# from pathlib import Path
# from tqdm import tqdm

# from config import DEFAULT_BUCKET, DEFAULT_REGION

# PROJECT_DIR = Path(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))



import sys # Python system library needed to load custom functions
import math # module with access to mathematical functions
import os # for changing the directory

import numpy as np  # for performing calculations on numerical arrays
import pandas as pd  # home of the DataFrame construct, _the_ most important object for Data Science

from datasets import load_dataset, Audio 

import librosa
import soundfile as sf

import torchaudio

import time


def white_noise(signal, noise_factor_min=0.1, noise_factor_max=0.4, noise_factor=None, seed=123): 

    np.random.seed(seed)
    noise = np.random.normal(0, signal.std(), signal.shape[0])
    if noise_factor is None:
        noise_factor = np.random.uniform(noise_factor_min, noise_factor_max)       
    signal_augmented = signal + noise * noise_factor
    return signal_augmented


def time_stretch(signal, stretch_rate_min=1.05, stretch_rate_max=1.2, stretch_rate=None, seed=123):    
    if (stretch_rate_min) < 1 & (stretch_rate_max > 1):
        raise ValueError("Both 'stretch_rate_min' and 'stretch_rate_max' must be either below 1 or above 1")
    if stretch_rate is None:
        np.random.seed(seed)
        stretch_rate = np.random.uniform(stretch_rate_min, stretch_rate_max)
    return librosa.effects.time_stretch(y=signal, rate=stretch_rate)


def pitch_scale(signal, sr, n_steps_min=1, n_steps_max=2, n_steps=None, seed=123):
    if n_steps is None:
        np.random.seed(seed)
        n_steps = np.random.uniform(n_steps_min, n_steps_max)    
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)  

def polarity_inversion(signal):
    return signal * -1


def random_gain(signal, gain_min=1.5, gain_max=3, gain=None, seed=123):
  
    if gain is None:  
        np.random.seed(seed) 
        gain = np.random.uniform(gain_min, gain_max)
    return signal * gain



def augmentation(signal, seed, sr):
    
    np.random.seed(seed) 
    aug_selection = np.sort(np.random.choice(5, size=np.random.randint(1,5), replace=False)+1)
    signal_aug = signal.copy()
    for i in np.sort(aug_selection):
       # np.random.seed(seed) 
        if i == 1:
            signal_aug = time_strech(signal=signal_aug, seed=seed) 
        if i == 2:
            signal_aug= pitch_scale(signal=signal_aug, sr=sr, seed=seed)  
        if i == 3:
            signal_aug = random_gain(signal=signal_aug, seed=seed)
        if i == 4:
            signal_aug = white_noise(signal=signal_aug, seed=seed)
        if i == 5:
            signal_aug = polarity_inversion(signal=signal_aug)            
    return  signal_aug   
            
    
    
def generate_augmented_data(train_data, metadata, path_out, fraction=0.5, seed=123):    
    
    sr = train_data[0]['audio']['sampling_rate']
    
    print('--------- data preprocessing stage ---------')
    t = time.time()
    path_train_data = [train_data[i]['audio']['path'] for i in range(train_data.shape[0])]
    path_train_data = np.array(path_train_data)
    if isinstance(metadata, str):
        metadata = pd.read_csv(metadata)
    df_audio = metadata[metadata.subset=='train']
    df_audio = df_audio.assign(path_full_raw = '/root/data/' + df_audio.path)
    df_audio = df_audio.assign(audio_rank_id = (df_audio.groupby('label')['file_name'].rank(method='first', na_option = 'bottom')-1).astype(int))

    df_audio_agg = df_audio.label.value_counts().reset_index().sort_values(by='label').reset_index(drop=True)
    count_max = df_audio_agg['count'].max()
    threshold = np.floor(count_max * fraction)
    df_audio_agg['gap'] = np.where((threshold - df_audio_agg['count']) > 0, (threshold - df_audio_agg['count']).astype(int), 0)
    df_audio_agg = df_audio_agg[df_audio_agg.gap > 0]

    label = []
    rank_id = []
    for l, c, g in zip(df_audio_agg['label'], df_audio_agg['count'], df_audio_agg['gap']):
        np.random.seed(seed) 
        ind = np.random.choice(c, size=g, replace=True)
        label.extend([l] * len(ind))
        rank_id.extend(ind)
    df_aug = pd.DataFrame(dict(label=label, audio_rank_id=rank_id))
    df_aug = df_aug.assign(rank_id = (df_aug.groupby(['label', 'audio_rank_id'])['audio_rank_id'].rank(method='first', na_option = 'bottom')-1).astype(int))
    
    df_aug_final = pd.merge(df_audio, df_aug, on=['label', 'audio_rank_id'], how='inner')
    df_aug_final = df_aug_final.assign(file_name = df_aug_final.file_name.apply(lambda x: x.split('.')[0]) + '_aug' + df_aug_final.rank_id.astype(str) + '.wav')    
    df_aug_final = df_aug_final.assign(path = 'data/train/' + df_aug_final.file_name)  
    print('Elapsed time:', time.time() - t)
    
    
    print('\n\n--------- augmentation stage ---------')    
    t = time.time()
    for row, (p, f) in enumerate(zip(df_aug_final.path_full_raw.iloc, df_aug_final.file_name.iloc)):
        i = int(np.where(path_train_data == p)[0][0])
        sig_aug =  augmentation(signal=train_data[i]['audio']['array'], seed=123, sr=sr)   
        
        df_aug_final.iat[row, df_aug_final.columns.get_loc('num_frames')] = sig_aug.shape[0]  
        df_aug_final.iat[row, df_aug_final.columns.get_loc('length')] = sig_aug.shape[0]/sr
        sf.write(path_out + f, sig_aug, sr)    
    metadata_aug = pd.concat([metadata, df_aug_final[['file_name', 'unique_file', 'path', 'species', 'label', 'subset', 'sample_rate', 'num_frames', 'length']]], axis=0)
    metadata_aug.to_csv(path_out + 'meta_data.csv', index=False)
    print('Elapsed time:', time.time() - t)    