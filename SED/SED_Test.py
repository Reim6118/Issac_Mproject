import argparse
import time
import torch

import tqdm
import SED as sed
import SED_model as sedm
import SED_get as getmodel2
import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from SED import *
from more_itertools import chunked 
from matplotlib import pyplot as plt

import tensorflow.keras
import tensorflow.math

##"Designing a Framework for Haptic Conversion of Visual Action Events within 
# Eye Gaze Range to Tactile Sensations: A Novel Approach for Enhancing Accessibility and Human-Computer Interaction"
#####Hapticizing Visual Action Events: A Framework for Converting Events Occurring within Eye Gaze Range to Tactile Sensations

file_count = 9  #Change heree when additional files
sr = 16000
n_mels=32 
# n_fft=1024 
# hop_length=512
time_resolution = 0.10
batch_size=4
samplerate = 16000
# csvpath = r"C:\Users\issac\Documents\ML\Badminton_sound\crop.csv"
# audiopath = r"C:\Users\issac\Documents\ML\Badminton_sound\crop"
csvpath = r"C:\Users\issac\Documents\ML\Badminton_sound\sound.csv"
audiopath = r"C:\Users\issac\Documents\ML\Badminton_sound\audio_wav"
# csvpath = r"C:\Users\issac\Documents\ML\Badminton_sound\test2.csv"
# audiopath = r"C:\Users\issac\Documents\ML\Badminton_sound\test2"
window_duration = 0.801 
window_length = int(window_duration / time_resolution)

def next_power_of_2(x):
    return 2**(math.ceil(math.log(x, 2)))

hop_length = int(time_resolution*samplerate)
n_fft = next_power_of_2(hop_length)

audiofiles,spec = sed.LoadAudio(audiopath,sr,n_mels,n_fft,hop_length)
data = sed.LoadCsv(csvpath)

l = sed.labeling(data, spec, time_resolution,file_count) #make the label to be continuous even data without event
print(l["1"].event.value_counts())  #find how many label of hit in all timeframe
fig, ax = plt.subplots(1, figsize=(20, 5))
# sed.plot_spectrogram(hop_length,samplerate,ax, spec[0], data[data['file'] == 1],l["1"])
# plt.show()

print("Array shape= ", l["1"].shape)
append_windows = pd.DataFrame()  #create empty dataframe
# append all the windows dataframe from each audio source file into one dataframe called append_windows
for i,audio in enumerate(audiofiles):
    print("i=",i,"audiofile=",audio)
    windows = pd.DataFrame({
        'spectrogram': sed.crop_windows(spec[i], frames=window_length, step=2),
        'labels': sed.crop_windows(l[str(i+1)].values.T, frames=window_length, step=2),
        'file': audiofiles[i],
    })
    append_windows = pd.concat([append_windows,windows],axis=0)


# sample and plot some windows to verify
append_windows['event'] = append_windows.labels.apply(lambda labels: np.any(labels, axis=-1)).astype(int)
append_windows[append_windows.event == True].head(5)
append_windows.groupby('event').sample(n=20).groupby('event').apply(sed.plot_windows,hop_length = hop_length, samplerate = samplerate,col_wrap=5, aspect=2, height=2)

######
# l is dictionary: 1,2,3,4 , data is dataframe, spec is list : 0,1,2,3, audiofiles is list: 0,1,2,3
######
splitData = sed.split_data(append_windows)
# plt.show()

###########################
#model1 ##################
###########################
import tensorflow.keras.backend as K
from keras.utils import custom_object_scope
from keras.models import load_model
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

with custom_object_scope( {'weighted_binary_crossentropy':weighted_binary_crossentropy}):

    model = load_model(r'C:\Users\issac\Documents\ML\Badminton_sound\model\7file(10000).h5')
model.summary()
  



from tqdm.keras import TqdmCallback

# # # Compute the spectral background across entire clip
# # # Used for spectral subtraction, a type of preprocessing/normalization technique that is often useful
Xm = np.expand_dims(np.mean(np.concatenate([s.T for s in splitData.spectrogram]), axis=0), -1)

# def get_XY(split):
#     # convenience to convert
#     d = splitData[splitData.split == split]
    
#     X = np.expand_dims(np.stack([(s-Xm).T for s in d.spectrogram]), -1)

#     Y = np.stack([ l.T for l in d.labels], axis=0)    
#     return X, Y

# train = get_XY(split='train')
# val = get_XY(split='val')

# test = get_XY(split='test')

# results = pd.DataFrame({
#     'split': [ 'test', 'train', 'val' ],
# })
# def get_metric(split):
#     X, Y = get_XY(split=split)
#     r = model.evaluate(x=X, y=Y, return_dict=True, verbose=False)
#     return pd.Series(r)

# e = results.split.apply(get_metric)
# results = pd.merge(results, e, right_index=True, left_index=True).set_index('split')

# from sklearn.metrics import PrecisionRecallDisplay

# fig, ax = plt.subplots(1)

# for split in results.reset_index().split.unique():
#     X, Y = get_XY(split)
    
#     y_true = Y
#     y_pred = model.predict(X, verbose=False)
    
#     y_true = np.any(y_true, axis=1)
#     y_pred = np.max(y_pred, axis=1)
    
#     PrecisionRecallDisplay.from_predictions(ax=ax, y_true=y_true, y_pred=y_pred, name=split)

# ax.axhline(0.9, ls='--', color='black', alpha=0.5)
# ax.axvline(0.9, ls='--', color='black', alpha=0.5)

def predict_spectrogram(model, spec,window_length):
    
    # prepare input data. NOTE: must match the training preparation in getXY
    window_hop = 1
    wins = sed.crop_windows(spec, frames=window_length, step=window_hop)       
    X = np.expand_dims(np.stack( [ (w-Xm).T for w in wins ]), -1)
    
    # make predictions on windows
    y = np.squeeze(model.predict(X, verbose=False))
    
    out = sed.merge_overlapped_predictions(y, window_hop=window_hop)

    return out


##目前只是用第一個file做結果的測試而已
predictions = predict_spectrogram(model,spec[5],window_length)
fig, ax = plt.subplots(1, figsize=(30, 5))
sed.plot_spectrogram(hop_length,samplerate,ax, spec[5],  data[data['file'] == 6],l["6"], predictions)
# model.save(r'C:\Users\issac\Documents\ML\Badminton_sound\model\6file(6000).h5')



plt.show()





#####################
########model2######
####################
# configs, server_cfg, train_cfg, feature_cfg = getmodel2.get_configs(config_dir=r"C:\Users\issac\Documents\ML\Yolov8\Code\SED\SED_config.yaml")

# device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

# train_cfg["device"] = device
# net,emanet= getmodel2.get_models(configs,train_cfg,False)
# outputs = net(train)
##他是用dataloader讀資料 我現在的data是keras的numpy形式 要轉成tensor之類的讓pytorch可以用