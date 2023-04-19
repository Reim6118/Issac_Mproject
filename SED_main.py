import SED as sed
from SED import plot_spectrogram, crop_windows
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
window_duration = 0.801 
window_length = int(window_duration / time_resolution)

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

windows = pd.DataFrame({
    'spectrogram': crop_windows(spec[0], frames=window_length, step=2),
    'labels': crop_windows(l["1"].values.T, frames=window_length, step=2),
    # 'file': audio_path,
})
print(windows)
# windows['event'] = windows.labels.apply(lambda labels: np.any(labels, axis=-1)).astype(int)
# windows[windows.event == True].head(5)
######
# l is dictionary: 1,2,3,4 , data is dataframe, spec is list : 0,1,2,3
######


# fig, ax = plt.subplots(1, figsize=(30, 5))
# plot_spectrogram(hop_length,samplerate,ax, spec[0], data[data['file'] == 1], l["1"])
# ax.set_xlim(0, 15)
# plt.show()



