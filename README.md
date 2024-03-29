# Research Objective
#### This project aims to create a system that can automatically generate haptic audio for a video based on events, the output haptic audio can then be directly use in manual authoring tools to improve efficiency of manual authoring task. While in this particular research, the system is only capable to automatic generate haptic feedbacks from scenario of  badminton matches. 

# System Overview

##### The system consist of two deep learning model (Object detection, Sound Event Detection) to seperately process the audio and visual content from a video. Both of data will be processed with machine learning models and algorithms to find the frame that have high possibility containing the event that can generate haptics, in this case, the frame when the racket and badminton contacts. After filtering out the frames that fulfil the requirement, the system will create a haptic audio based on those frame with the haptic sensation directily recorded from a real badminton racket.

<img width="1325" alt="image" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/d1ef34a0-f65d-476d-a886-277db9954340">

## Processing Visual Content

##### The video will be first normalized and turned into gray scale. It will then pass through a subtraction algorithm to calculate the motion saliency, which indicates the moving part / pixels that is different from the previous frame. 
<img width="1325" alt="Screenshot 2024-02-26 at 15 49 37" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/5e5235de-e85d-4f6d-a365-91701421af17">

##### The processed frames will then be passed into an object detection model trained using transfer learning from Yolov8 (yolov8n.pt).

<img width="612" alt="image" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/7cd8dadf-b972-4086-a569-3d65e112995f">

##### The output frame from the model is shown on the picture on the right, which is a cropped mask with only the badminton that is identified. The position of the badminton will be recorded and pass to another algorithm to determine the possibility of racket contacting with the shuttle ball. 
<img width="1309" alt="Screenshot 2024-02-26 at 15 54 08" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/38fbd42e-a90f-4316-a7da-9ee305fab51a">

## Processing Audio Content

##### The audio data is first converted into a spectrogram before divided into smaller segments, it is then passed into a Sed-CRNN model, which is a sound event detection model trained based on convolution recurrent network. This model predicts the likelihood of each frame containing the sound of badminton shuttle being hit by a racket. Although the predictions regarding time boundary may not be precise enough to directly determine the exact frames with events, the frames with possibility exceeding 50 percent is marked and used to filter frames after combining with the output from object detection model.
<img width="1309" alt="Screenshot" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/256116ac-4b8c-4e7d-bd9d-e68c38769855">


## Combination of two models' output

##### With a comparison between the output frames from both visual and audio components, the frames that have overlapping marks are identified. Furthermore, three frames preceding and following each iedentified frames are also considered. Pre-recorded audio that represents the haptic sensation of the impact between racket and shuttle ball is added to those frames. The final output will be a quad-channel mp4 file, with the first two channels containing the original audio, while the latter two channels accommodate the hatic audio.


<!-- #### Interface of Ableton Live during manual annotation task 
![image](https://github.com/Reim6118/Issac_Mproject/assets/32570797/829c3795-cb9f-4c09-a5ee-ff0db78eb802) -->

## Final Output

##### A comparison between manual authored and automatic generated haptic feedback for a same piece of badminton match video is shown below. In this particular video, the automatic audio generation achieved an accuracy of 16 correct annotations out of 22 hits, correct annotations indicates the automatic generated haptic fells between +-25ms compare to manual authored haptic feedbacks. Note that the manual authoring haptic can also vary between experiment participants / haptic authors, however it is observed that participants cannot precisely tell the difference between haptic feedbacks that is provided +-50ms earlier/later, even around +-80ms.
<img width="523" alt="image" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/6b524ff8-4f3e-4020-950e-d993604e6538">

#### 3D printed artifacts with solenoid actuators embded in is provided to participants to directly feel the haptic feedback from videos.
<img width="385" alt="Screenshot2" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/e95251a8-a004-4779-b9c9-93060973b566">

#### Please refer to my thesis for a more detailed explanation. https://koara.lib.keio.ac.jp/xoonips/modules/xoonips/detail.php?koara_id=KO40001001-00002023-1014





















