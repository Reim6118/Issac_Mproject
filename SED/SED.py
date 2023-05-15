import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from more_itertools import chunked 
from matplotlib import pyplot as plt
from pydub import AudioSegment
import ffmpeg
import subprocess

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

class VideoReader:
    def __init__(self, Path):
        self.path = Path
        self.video = mp.VideoFileClip(self.path)
        self.audio = self.video.audio
    def video(self):
        return self.video
    def audio(self):
        return self.audio

import moviepy.editor as mp
def LoadFromVid(path,Sr,mels,fft,hop_length): #0,1,2,3
    # audio_files = lb.util.find_files(path)
    video = VideoReader(path)
    # video = mp.VideoFileClip(path)
    # audio = video.audio
    audio = video.audio
    temp = "Combine_test/temp.wav"
    audio.write_audiofile(temp)
    
    y, sr = lb.load(temp, sr=16000)
    Spec = lb.feature.melspectrogram(y=y, sr=Sr, n_mels=mels,n_fft=fft,hop_length=hop_length)
    Sdb = lb.power_to_db(Spec, ref=np.max)
    print("SDB shape = ",Sdb.shape)
    return temp,Sdb

# def make_sound(df):
#     t = np.linspace(0, duration, int(audio.fps * duration))
#     return (np.sin(2*np.pi*440*t/44100)).astype(np.float32)
    # if df['sound_data'].iloc[t]:
    #     return (np.sin(2*np.pi*440*t/44100)).astype(np.float32)
    # else:
    #     return np.array([0, 0], dtype=np.float32)
    
from moviepy.editor import *
# from pydub import AudioSegment
# def AddAudioChannel(path,sounddf ):
#     video = VideoFileClip(path)
#     audio = video.audio
#     # video_file = video.video
#     silenceaudio = r'C:\Users\issac\Documents\ML\Combine_test\silence.wav'
#     audiofile = r'C:\Users\issac\Documents\ML\Combine_test\500ms.wav'
#     add_audio = AudioFileClip(audiofile)
#     blank_image = ImageClip(path).set_duration(video.duration)
#     # silence_frame = AudioSegment.silent(duration=1000/add_audio.fps, frame_rate=add_audio.fps)
#     silence_audio = AudioFileClip(silenceaudio)
#     for index, row in sounddf.iterrows():
#         frame_number = row['Frame']
#         if row['Change_Point_X'] == 'Changed_X':
#             # video_file = video_file.set_audio(video_file.audio.set_audio_frame(add_audio.audio.to_audio_clip(), 3, frame_number))
#             audio_frame = add_audio.get_frame(frame_number / add_audio.fps)
#             # video.audio[frame_number] = audio_frame[:,:,None]
#             # video_file = video.audio.set_make_frame(audio_frame, idx=3, t=frame_number / video.fps)
#         else:
#             # silence_clip = mp.AudioClip(duration=1/30.0, fps=44100)
#             # video_file = video_file.set_audio(video_file.audio.set_audio_frame(silence_clip, 3, frame_number))
#             # silence_clip = mp.AudioClip(duration=1/30.0, fps=44100)
#             # audio_frame = silence_clip.make_audio_frame(0)
#             audio_frame = silence_audio.get_frame(frame_number / add_audio.fps)
#             video.audio[frame_number] = audio_frame[:,:,None]
#             # video_file = video.audio.set_make_frame(audio_frame, idx=3, t=frame_number / video.fps)

#     video.write_videofile('addchannel.mp4')
#     return

def Create_Audio_Sounddf(sounddf):
    silenceaudio = r'C:\Users\issac\Documents\ML\Combine_test\silence.wav'
    audiofile = r'C:\Users\issac\Documents\ML\Combine_test\badminton.mp3'
    sample_rate = 44100  # Sample rate (Hz)
    duration = 1/30
    frames = int(len(sounddf)* duration * sample_rate)  # Number of frames
    audio_data = np.zeros(frames )  
    
    for i, row in sounddf.iterrows():
        frame_start = i * int(sample_rate * duration)
        frame_end = (i + 3) * int(sample_rate * duration)

        # if row['Change_Point_X'] == 'Changed_X' or row['Change_Point_Y'] == 'Changed_Y':
        if row['Change_Point_X'] == 'Changed_X':
            #在這邊加不同的羽毛球音頻，然後讀db來判斷，寫幾個if，>50可以用殺球40-50中間<40小
            audio, _ = sf.read(audiofile)
            audio = np.reshape(audio,(-1))
            audio_data[frame_start:frame_end] = audio[:frame_end-frame_start]
        elif audio_data[frame_start] == None:
            audio, _ = sf.read(silenceaudio)           
            audio = np.reshape(audio,(-1))
            audio_data[frame_start:frame_end] = audio[:frame_end-frame_start] 
        # else:    
        #     audio, _ = sf.read(silenceaudio)           
        #     audio = np.reshape(audio,(-1))
        #     audio_data[frame_start:frame_end] = audio[:frame_end-frame_start]
    sf.write(r'C:\Users\issac\Documents\ML\Combine_test\sounddf_output.wav', audio_data, sample_rate)
    return

def calculate_dB_frame(frame, ref_level=1e-10):
    spectrogram = lb.stft(frame)
    rms = lb.feature.rms(S=spectrogram)
    dB = lb.amplitude_to_db(rms,ref=ref_level)
    avg_dB = np.mean(dB)
    return avg_dB
def caluculate_db(path,df):
    video_file = path
    sample_rate = 44100  # Sample rate (Hz)
    duration = 1/30
    db_values = []
    audio, sr = lb.load(video_file, sr=44100)
    for i, row in df.iterrows():
        frame_start = i * int(sample_rate * duration)
        frame_end = (i + 1) * int(sample_rate * duration)
        # audio = np.reshape(audio,(-1))
        db_value = calculate_dB_frame(audio[frame_start:frame_end])
        db_values.append(db_value)
    # dB_values = [calculate_dB_frame(audio[i:i+sr]) for i in range(0, len(audio), sr)]
    df['dB'] = db_values
    return df

def separate_video_and_audio(path):
    video_file = path
    output_video = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_video.mp4"
    output_audio1 = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_audio1.aac"
    output_audio2 = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_audio2.aac"

    # Separate video from the input file
    video_command = [
        "ffmpeg",
        "-i", video_file,
        "-c:v", "copy",
        "-an",  # Disable audio
        '-y',
        output_video
   ]
    # Separate audio channels from the input file
    audio_command1 = [
        "ffmpeg",
        "-i", video_file,
        "-map", "0:a:0",  # Select audio channel 1
        "-c:a", "copy",
        '-y',
        output_audio1
    ]
    audio_command2 = [
        "ffmpeg",
        "-i", video_file,
        "-map", "0:a:1",  # Select audio channel 2
        "-c:a", "copy",
        '-y',
        output_audio2
    ]
    # Execute the video and audio separation commands
    subprocess.run(video_command)
    subprocess.run(audio_command1)
    subprocess.run(audio_command2)
import subprocess




def EncodeAudioChannel(path ):
    original_video = path
      
    video = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_video.mp4"
    audio1 = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_audio1.aac"
    audio_originalL = r"C:\Users\issac\Documents\ML\Combine_test\separate\left.wav"
    audio_originalR = r"C:\Users\issac\Documents\ML\Combine_test\separate\right.wav"
    # audio2 = r'C:\Users\issac\Documents\ML\Combine_test\sounddf_output.wav'
    haptic_audio = r'C:\Users\issac\Documents\ML\Combine_test\sounddf_output.wav'  
    audio_output = r"C:\Users\issac\Documents\ML\Combine_test\output\audio_output.wav"
    
    #Seperate audio and video first then merge
    separate_video_and_audio(original_video)
    #Split the stereo into two mono
    Split_Stereo(audio1)
   
    ##############mine############
    ffmpeg_cmd = [
    'ffmpeg',
    '-i', audio_originalL,
    '-i', audio_originalR,
    '-i', haptic_audio,
    '-i', haptic_audio,
    '-filter_complex', '[0:a][1:a][2:a][3:a]join=inputs=4:channel_layout=quad[a]',
    '-map', '[a]',
    '-y',
    audio_output
]
    subprocess.run(ffmpeg_cmd)
    
    return

def Split_Stereo(path):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', path,
        '-filter_complex', '[0:a]channelsplit=channel_layout=stereo[left][right]',
        '-map', '[left]','-y', r"C:\Users\issac\Documents\ML\Combine_test\separate\left.wav",
        '-map', '[right]','-y', r"C:\Users\issac\Documents\ML\Combine_test\separate\right.wav"
    ]
    subprocess.run(ffmpeg_cmd)
    return 

def Combine_Vid_Audio():
    output = r"C:\Users\issac\Documents\ML\Combine_test\output\output2.mp4"
    split_vid = r"C:\Users\issac\Documents\ML\Combine_test\separate\output_video.mp4"
    audio = r"C:\Users\issac\Documents\ML\Combine_test\output\audio_output.wav"
    ffmpeg_cmd = ['ffmpeg ',
                  '-i', split_vid,
                  '-i', audio,
                  '-c:v','copy',
                #   '-c:a', 'aac', 
                  '-map',' 0:v:0',
                  '-map', '1:a:0', 
                  '-shortest', 
                  output]
    subprocess.run(ffmpeg_cmd)
    return

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
        ##original##
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
        ###mine randomize##
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
    
def split_data2(data, val_size=0.25, test_size=0.25, random_state=3, column='split'):

   
    data = data.sample(frac=1).reset_index(drop = True)
    val_stop = 0
    test_idx = data.index[val_stop:-1]

    data = data.copy()
    data.loc[test_idx, column] = 'test'
    return data
def merge_overlapped_predictions(window_predictions, window_hop):
    
    # flatten the predictions from overlapped windows
    predictions = []
    for win_no, win_pred in enumerate(window_predictions):
        win_start = window_hop * win_no
        for frame_no, p in enumerate(win_pred):
            s = {
                'frame': win_start + frame_no,
                'probability': p,
            }
        
            predictions.append(s)
        
    df = pd.DataFrame.from_records(predictions)
    df['time'] = pd.to_timedelta(df['frame'] * 0.10, unit='s')
    df = df.drop(columns=['frame'])
    
    # merge predictions from multiple windows 
    out = df.groupby('time').median()
    return out

def events_from_predictions(pred, threshold=0.1, label='Hit', event_duration_max=1.0,fps =30):
    import copy
    
    event_duration_max = pd.Timedelta(event_duration_max, unit='s')
    
    events = []
    inside_event = False
    event = {
        'start': None,
        'end': None,
    }
    
    for t, r in pred.iterrows():
        p = r['probability']

        # basic state machine for producing events
        if not inside_event and p > threshold:
            event['start'] = int(t.total_seconds()  *fps)  #Modify here to calculate which frame it is in
            inside_event = True
            
        elif inside_event and ((p < threshold) or ((t - pd.Timedelta(seconds=event['start']/fps)) > event_duration_max)):
            event['end'] = int(t.total_seconds() *fps)
            events.append(copy.copy(event))
            
            inside_event = False
            event['start'] = None
            event['end'] = None
        else:
            pass
    
    if len(events):
        df = pd.DataFrame.from_records(events)
    else:
        df = pd.DataFrame([], columns=['start', 'end'], dtype='timedelta64[ns]')
    df['label'] = label
    return df

def predict_spectrogram(model, spec,window_length ,Xm):
    
    # prepare input data. NOTE: must match the training preparation in getXY
    window_hop = 1
    wins = crop_windows(spec, frames=window_length, step=window_hop)       
    X = np.expand_dims(np.stack( [ (w-Xm).T for w in wins ]), -1)
    
    # make predictions on windows
    y = np.squeeze(model.predict(X, verbose=False))
    
    out = merge_overlapped_predictions(y, window_hop=window_hop)

    return out
import tensorflow.keras.backend as K

def weighted_binary_crossentropy(zero_weight, one_weight):
    """
    Loss with support for specifying class weights
    """
    
    
    def weighted_binary_crossentropy(y_true, y_pred):

        # standard cross entropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # apply weighting
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy