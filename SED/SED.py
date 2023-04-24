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
    return audio_files, Sdb_List



def labeling(originaldata, length, time_resolution, i):
    """
    Create a continious vector for the event labels that matches the time format of our spectrogram
    
    Assumes that no annotated event means nothing occurred.
    """
    dataframes = {}
    for i in range(1,i+1):       
        data = originaldata[originaldata['file'] == i]
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
        print("i = ", i)
    
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
            
def crop_windows(arr, frames, pad_value=0.0, overlap=0.5, step=None):
    if step is None:
        step = int(frames * (1-overlap))
        
    windows = []
    index = []
        
    width, length = arr.shape
    
    for start_idx in range(0, length, step):
        end_idx = min(start_idx + frames, length)

        # create emmpty
        win = np.full((width, frames), pad_value, dtype=float)
        # fill with data
        win[:, 0:end_idx-start_idx] = arr[:,start_idx:end_idx]  # crop a fix frame from start index till start index +n

        windows.append(win)
        index.append(start_idx)

    s = pd.Series(windows, index=index)
    s.index.name = 'start_index'
    return s


  
def plot_windows( wins,hop_length, samplerate, col_wrap=None, height=4, aspect=1):
    specs = wins.spectrogram
    
    nrow = 1
    ncol = len(specs)
    if col_wrap is not None:
        nrow = int(np.ceil(ncol / col_wrap))
        ncol = col_wrap

    fig_height = height * nrow
    fig_width = height * aspect * ncol
    fig, axs = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(fig_width, fig_height))
    axs = np.array(axs).flatten()
    
    fig.suptitle(specs.name)
    for ax, s, l in zip(axs, specs, wins.labels):
    
        l = np.squeeze(l)
        ll = pd.DataFrame({
            'event': l,
            'time': pd.to_timedelta(np.arange(l.shape[0])*hop_length/samplerate, unit='s'),
        })

        plot_spectrogram(hop_length,samplerate,ax, s, label_activations=ll)

def split_data(data, val_size=0.25, test_size=0.25, random_state=3, column='split'):
    """
    Split DataFrame into 3 non-overlapping parts: train,val,test
    with specified proportions
    
    Returns a new DataFrame with the rows marked by the assigned split in @column
    """

    data = data.sample(frac=1).reset_index(drop = True)
    train_size = (1.0 - val_size - test_size)

    train_stop = int(len(data) * train_size)
    val_stop = train_stop + int(len(data)*val_size)
    
    train_idx = data.index[0:train_stop]
    val_idx = data.index[train_stop:val_stop]
    test_idx = data.index[val_stop:-1]
    
    data = data.copy()
    data.loc[train_idx, column] = 'train'
    data.loc[val_idx, column] = 'val'
    data.loc[test_idx, column] = 'test'

    # train_size = (1.0 - val_size - test_size)

    # train_stop = int(len(data) * train_size)
    # val_stop = train_stop + int(len(data)*val_size)
    
    
    # train_idx = data.sample(n=train_stop)       # random all * 0.5
    # print("train stop" , train_stop)
    # print("Train index=",len(train_idx))
    # left = data.drop(train_idx.index)  
    # print("len left",len(left))         # left half                
    # val_idx = left.sample(n = int(len(left)*0.5))      #
    # test_idx= left.drop(val_idx.index)
    
    # train_idx[column] = 'train'
    # val_idx[column] = 'val'
    # test_idx[column] = 'test'
    # data = pd.concat([train_idx,val_idx,test_idx])

    

    return data
    