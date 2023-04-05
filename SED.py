import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from more_itertools import chunked 
from matplotlib import pyplot as plt


#samplerate
sr = 16000
n_mels=32 
n_fft=128
hop_length=512
batch_size=4
csvpath = r"C:\Users\issac\Documents\ML\Badminton_sound\sound.csv"
audiopath = r"C:\Users\issac\Documents\ML\Badminton_sound\audio_wav"
# def load(path):
#     df = pd.read_csv(path, header=None)
#     df.columns = ['Event', 'Start', 'End','File']
#     # df['Duration'] = pd.to_numeric(df['End']) - pd.to_numeric(df['Start'])
#     df['duration'] = df['End'].astype(np.float32) - df['Start'].astype(np.float32)
#     return df

def LoadCsv(path):
    df = pd.read_csv(path, header=None, skiprows=1, names=['event', 'start', 'end', 'file'])
    df['duration'] = df['end'].astype(np.float16) - df['start'].astype(np.float16)
    return df

def LoadAudio(path,Sr,mels,fft,hop_length):
    audio_files = lb.util.find_files(path)
    print(audio_files)
    Sdb_List= []
    for audio_file in audio_files:
        y, sr = sf.read(audio_file)
        Spec = lb.feature.melspectrogram(y=y, sr=Sr, n_mels=mels,n_fft=fft,hop_length=hop_length)
        Sdb = lb.power_to_db(Spec, ref=np.max)
        print("SDB shape = ",Sdb.shape)
    # y, sr = sf.read(path)
    
    # Spec = lb.feature.melspectrogram(y=y, sr=Sr, n_mels=mels,n_fft=fft,hop_length=hop_length)
    # Sdb = lb.power_to_db(Spec, ref=np.max)
    # print("SDB shape = ",Sdb.shape)
    # lb.display.specshow(Sdb, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.show()
    return



LoadAudio(audiopath,sr,n_mels,n_fft,hop_length)
data = LoadCsv(csvpath)
plt.hist(data['duration'],width=0.10, range=(0, 1.0))
plt.show()
# Extract the data from the identified columns
print(data)
