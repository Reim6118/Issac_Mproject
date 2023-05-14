from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import SED.SED as sed
import math
import os.path
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from SED.SED import *
from more_itertools import chunked 
from keras.utils import custom_object_scope
from keras.models import load_model
import tensorflow.keras
import tensorflow.math

#####################################################################################
##########################################Saliency###################################
#####################################################################################
Input_video = r"C:\Users\issac\Documents\ML\Combine_test\Input_video\badminton2.mp4"
cap = cv2.VideoCapture(Input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS,30)

threshold = 3
print("FPS:", fps)
saliency = None
count = 0
name = r"C:\Users\issac\Documents\ML\Combine_test\output1"
# create a motion saliency object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

ret, frame = cap.read()
height, width, _ = frame.shape
out = cv2.VideoWriter(name+'.mp4',fourcc,fps,(width, height),0)
out.write(frame)
while True:
    # read a frame from the input video
    
    print("loading",end='\r')
    ret, frame = cap.read()
    if not ret:
        break
    if saliency is None:
        print(frame.shape[1],',',frame.shape[0])
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        print(frame.shape[1],',',frame.shape[0])
        saliency.init()
        saliency_update = False

    print("Calculating Saliency",end='\r')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    meannn = np.mean(saliencyMap)
    # if meannn> threshold and meannn < 200:
    #     out.write(saliencyMap)
    # if not saliency_update:
    #     saliency_update = True
    out.write(saliencyMap)
    
    
    print("Calculating Saliency...",end='\r')
    # out.write(saliencyMap)
    
    

out.release()
cap.release()
# #####################################################################################
# ##########################################YOLO########################################
# #####################################################################################

model = YOLO("Yolov8/OnlyBadminton.pt")
count =0
frame_num=0
df = pd.DataFrame(columns=['Frame', 'Blank','Center_X','Center_Y','Sec'])
results = model.predict(source=r"C:\Users\issac\Documents\ML\Combine_test\output1.mp4", save =False, save_crop = False) # Display preds. Accepts all YOLO predict arguments
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\issac\Documents\ML\Combine_test\crop_mask'+'.mp4',fourcc,fps,(width, height))
# badmintonout = cv2.VideoWriter(r'C:\Users\issac\Documents\ML\Combine_test\badmintoncrop_mask'+'.mp4',fourcc,fps,(width, height))

for result in results:
    isBlank = "Blank"
    isBadminton = "Badminton"
    ret, frame = cap.read()
    mask = np.ones(result.orig_img.shape,dtype= result.orig_img.dtype)
    # badmintonmask = np.ones(result.orig_img.shape,dtype= result.orig_img.dtype)

    mask[:,:] = [255,255,255] 
    # badmintonmask[:,:] = [255,255,255]

    for bbox in result.boxes.xyxy:
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        crop_image=result.orig_img[int(y1):int(y2),int(x1):int(x2)]
        #paste the crop image to the blank mask
        mask[int(y1):int(y2),int(x1):int(x2)] = crop_image
        meancrop = np.mean(crop_image)
        #Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # if meancrop < 40:
            # badmintonmask[int(y1):int(y2),int(x1):int(x2)] = crop_image
    maskmean = np.mean(mask)
    # print(maskmean)
    #
    if maskmean ==255.0 :
        df.loc[frame_num] = [frame_num, isBlank,0,0,frame_num/fps]
    else:
        df.loc[frame_num] = [frame_num,isBadminton, center_x,center_y,frame_num/fps]
    frame_num+=1

    
    out.write(mask)
    # badmintonout.write(badmintonmask)


window_size_ascend = 3
###########################Calculate X#################################
mask_X = (df['Center_X'] == df['Center_X'].shift()) | (df['Center_X'] == 0)
df.loc[mask_X, 'Center_X'] = None
df['Center_X'].fillna(method='ffill', inplace=True)
diff_X = df['Center_X'].diff(periods=window_size_ascend)
threshold = 3
df['Big change X'] = ''
df['Direction_X'] = None
df['ShiftedDirection_x'] = ''
df['Change_Point_X'] = ''
df.loc[diff_X > 0, 'Direction_X'] = 'Right'
df.loc[diff_X < 0, 'Direction_X'] = 'Left'
# fill the direction for the zero rows
last_nonzero_direction_X = None
for i, row in df.iterrows():
    if pd.isna(row['Direction_X']):
        df.at[i, 'Direction_X'] = last_nonzero_direction_X
    else:
        last_nonzero_direction_X = row['Direction_X']
df['ShiftedDirection_X'] = df['Direction_X'].shift(1)
change_point_X = df[((df['Direction_X'] == 'Left') & (df['ShiftedDirection_X'] == 'Right')) | ((df['Direction_X'] == 'Right')& (df['ShiftedDirection_X'] == 'Left')) | (((df['Center_X'] - df['Center_X'].shift(1)).abs()) > 50)].index.tolist()
df.loc[change_point_X,'Change_Point_X'] = 'Changed_X'


###########################Calculate Y#################################
mask_Y = (df['Center_Y'] == df['Center_Y'].shift()) | (df['Center_Y'] == 0)
df.loc[mask_Y, 'Center_Y'] = None
df['Center_Y'].fillna(method='ffill', inplace=True)
diff_X = df['Center_Y'].diff(periods=window_size_ascend)
threshold = 3
df['Big change Y'] = ''
df['Direction_Y'] = None
df['ShiftedDirection_Y'] = ''
df['Change_Point_Y'] = ''
df.loc[diff_X > 0, 'Direction_Y'] = 'Up'
df.loc[diff_X < 0, 'Direction_Y'] = 'Down'
# fill the direction for the zero rows
last_nonzero_direction_Y = None
for i, row in df.iterrows():
    if pd.isna(row['Direction_Y']):
        df.at[i, 'Direction_Y'] = last_nonzero_direction_Y
    else:
        last_nonzero_direction_X = row['Direction_Y']
df['ShiftedDirection_Y'] = df['Direction_Y'].shift(1)
change_point_Y = df[((df['Direction_Y'] == 'Down') & (df['ShiftedDirection_Y'] == 'Up')) | ((df['Direction_Y'] == 'Up')& (df['ShiftedDirection_Y'] == 'Down')) | (((df['Center_Y'] - df['Center_Y'].shift(1)).abs()) > 50)].index.tolist()
df.loc[change_point_Y,'Change_Point_Y'] = 'Changed_Y'

# df.loc[drastic_changes_Y,'Big change Y'] = 'True'
# df.loc[drastic_changes_X,'Big change X'] = 'True'

#降低或升高超過一定幅度，可能100也要changed x，

print(df)  
        


out.release()

#####################################################################################
##########################################SED########################################
#####################################################################################



file_count = 1  #Change heree when additional files
sr = 16000
n_mels=32 
datalist =[]
startlist =[]
endlist = []
durationlist = []
# n_fft=1024 
# hop_length=256
time_resolution = 0.10
batch_size=4
samplerate = 16000
window_duration = 0.801 
window_length = int(window_duration / time_resolution)

def next_power_of_2(x):
    return 2**(math.ceil(math.log(x, 2)))

hop_length = int(time_resolution*samplerate)
n_fft = next_power_of_2(hop_length)

audiofile,spec = sed.LoadFromVid(Input_video,sr,n_mels,n_fft,hop_length)
fig, ax = plt.subplots(1, figsize=(20, 5))
plot_spectrogram(hop_length,samplerate,ax,spec)
append_windows = pd.DataFrame()  #create empty dataframe
# append all the windows dataframe from each audio source file into one dataframe called append_windows  
windows = pd.DataFrame({
    'spectrogram': sed.crop_windows(spec, frames=window_length, step=2),  
    'file': audiofile,
})
append_windows = pd.concat([append_windows,windows],axis=0)
splitData = sed.split_data2(append_windows)

# Load model with customize crossentropy
with custom_object_scope( {'weighted_binary_crossentropy':weighted_binary_crossentropy}):

    model = load_model(r'C:\Users\issac\Documents\ML\Badminton_sound\model\onlyfinal(600).h5')
model.summary()
  
Xm = np.expand_dims(np.mean(np.concatenate([s.T for s in splitData.spectrogram]), axis=0), -1)
predictions = predict_spectrogram(model,spec,window_length,Xm)
fig, ax = plt.subplots(1, figsize=(30, 5))
sed.plot_spectrogram(hop_length,samplerate,ax, spec, predictions = predictions)
annotate = events_from_predictions(predictions)     
for index, row in annotate.iterrows():
    datalist.append(list(row))
startlist = [row[0] for row in datalist]
endlist = [min(row[1],frame_num-1) for row in datalist]
for i in range(len(startlist)):
    templist = [x for x in range(startlist[i],endlist[i]+1)]
    durationlist.append(templist)
# df['Sound_Detect'] = 'No'
# df['Haptic'] = 'No'
# for i in range(len(durationlist)):
    # df.loc[durationlist[i],'Sound_Detect'] = 'Hit'
# for index, row in df.iterrows():
#     if row['Sound_Detect'] == 'Hit' and row['Blank'] == 'Blank':
#         df.loc[index,'Haptic'] = 'Yes'
sounddf = df[['Frame','Change_Point_X']].copy()
Create_Audio_Sounddf(sounddf)
AddAudioChannel(Input_video)
plt.show()



##還要算羽毛球的位置， 看是上半部還是下半部，分別把haptic的聲音加到兩個不同的channel