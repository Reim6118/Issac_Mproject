import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from more_itertools import chunked 
from matplotlib import pyplot as plt


#samplerate

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

def LoadAudio(path,Sr,mels,fft,hop_length): #0,1,2,3
    audio_files = lb.util.find_files(path)
    print(audio_files)
    Sdb_List= []
    for audio_file in audio_files:
        y, sr = lb.load(audio_file)
        Spec = lb.feature.melspectrogram(y=y, sr=Sr, n_mels=mels,n_fft=fft,hop_length=hop_length)
        Sdb = lb.power_to_db(Spec, ref=np.max)
        print("SDB shape = ",Sdb.shape)
        Sdb_List.append(Sdb)
    return Sdb_List



def make_continious_labels(data, length, time_resolution, i):
    """
    Create a continious vector for the event labels that matches the time format of our spectrogram
    
    Assumes that no annotated event means nothing occurred.
    """
    dataframes = {}
    for i in range(1,i+1):       
        data = data[data['file'] == i]
        freq = pd.Timedelta(seconds=time_resolution)
        
        # Create empty covering entire spectrogram
        duration = length[i-1].shape[1] * time_resolution
        ix = pd.timedelta_range(start=pd.Timedelta(seconds=0.0),
                        end=pd.Timedelta(seconds=duration),
                        freq=freq,
                        closed='left',
        )
        ix.name = 'time'
        df = pd.DataFrame({}, index=ix)
        assert len(df) == length[i-1].shape[1], (len(df), length[i-1].shape[1])
        df["event"] = 0
        
        # fill in event data
        for start, end in zip(data['start'], data['end']):
            s = pd.Timedelta(start, unit='s')   #create timedelta object for better manipulate in pandas dataframe
            e = pd.Timedelta(end, unit='s')

            # XXX: focus just on onsets
            #e = s + pd.Timedelta(0.100, unit='s') 
            
            match = df.loc[s:e]
            df.loc[s:e, "event"] = 1
        dataframes[str(i)] = df
        print(dataframes)
        # print("i = ", i)
    
    return dataframes

def plot_spectrogram(hop_length,samplerate,ax, spec, events=None, label_activations=None, predictions=None):
    events_lw = 1.5
    
    # Plot spectrogram
    lb.display.specshow(ax=ax, data=spec, hop_length=hop_length, x_axis='time', y_axis='mel', sr=samplerate)

    # Plot events
    if events is not None:
        for start, end in zip(events.start, events.end):
            ax.axvspan(start, end, alpha=0.2, color='yellow')
            ax.axvline(start, alpha=0.7, color='yellow', ls='--', lw=events_lw)
            ax.axvline(end, alpha=0.8, color='green', ls='--', lw=events_lw)

    label_ax = ax.twinx()
    
    # Plot event activations
    if label_activations is not None:
        a = label_activations.reset_index()
        a['time'] = a['time'].dt.total_seconds()
        label_ax.step(a['time'], a['event'], color='green', alpha=0.9, lw=2.0)

    # Plot model predictions
    if predictions is not None:
        p = predictions.reset_index()
        p['time'] = p['time'].dt.total_seconds()
        label_ax.step(p['time'], p['probability'], color='blue', alpha=0.9, lw=3.0)
            
        label_ax.axhline(0.5, ls='--', color='black', alpha=0.5, lw=2.0)
            

 