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


<img width="385" alt="Screenshot2" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/e95251a8-a004-4779-b9c9-93060973b566">

![image](https://github.com/Reim6118/Issac_Mproject/assets/32570797/829c3795-cb9f-4c09-a5ee-ff0db78eb802)


<img width="523" alt="image" src="https://github.com/Reim6118/Issac_Mproject/assets/32570797/6b524ff8-4f3e-4020-950e-d993604e6538">

