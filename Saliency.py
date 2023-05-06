import cv2
import numpy as np
# create a VideoCapture object to read the input video
#cap = cv2.VideoCapture(r"C:\Users\issac\Documents\badminton1.mp4")
cap = cv2.VideoCapture(r"C:\Users\issac\Documents\ML\Combine_test\badminton2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
saliency = None
count = 0
name = "output3"
# create a motion saliency object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name+'.mp4',fourcc,30.0,(1920, 1080),0)

while True:
    # read a frame from the input video
    
 
    ret, frame = cap.read()
    if not ret:
        break
    if saliency is None:
        print(frame.shape[1],',',frame.shape[0])
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        print(frame.shape[1],',',frame.shape[0])
        saliency.init()
    #if np.sum( np.absolute(frame-frame0) )/np.size(frame) > threshold: 設treshold
        
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    # show the output
    # Save each frame into jpg
    if ret == True:
        count+=1
        print("count=",count)
        cv2.imwrite('C:/Users/issac/Documents/ML/Yolov8/Saliency_Badminton2/frame_'+str(count)+'.jpg', saliencyMap)
    # out.write(saliencyMap)
    # cv2.imshow('Motion Saliency Map', saliencyMap)
    
    
    if cv2.waitKey(1) == ord('q'):
        break

# release the VideoCapture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
