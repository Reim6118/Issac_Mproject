import SED as sed
from SED import plot_spectrogram
import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from more_itertools import chunked 
from matplotlib import pyplot as plt



file_count = 4
sr = 16000
n_mels=32 
# n_fft=1024 
# hop_length=512
time_resolution = 0.10
batch_size=4
samplerate = 16000
csvpath = r"C:\Users\issac\Documents\ML\Badminton_sound\sound.csv"
audiopath = r"C:\Users\issac\Documents\ML\Badminton_sound\audio_wav"


def next_power_of_2(x):
    return 2**(math.ceil(math.log(x, 2)))

hop_length = int(time_resolution*samplerate)
n_fft = next_power_of_2(hop_length)

spec = sed.LoadAudio(audiopath,sr,n_mels,n_fft,hop_length)
data = sed.LoadCsv(csvpath)

l = sed.labeling(data, spec, time_resolution,file_count) #make the label to be continuous even data without event
print(l["1"].event.value_counts())  #find how many label of hit in all timeframe
fig, ax = plt.subplots(1, figsize=(20, 5))
plot_spectrogram(hop_length,samplerate,ax, spec[0], data[data['file'] == 1],l["1"])
plt.show()

print("Array shape= ", l["1"].shape)
######
# l is dictionary, data is dataframe, spec is list
######


# fig, ax = plt.subplots(1, figsize=(30, 5))
# plot_spectrogram(hop_length,samplerate,ax, spec[0], data[data['file'] == 1], l["1"])
# ax.set_xlim(0, 15)
# plt.show()



