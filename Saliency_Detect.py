import cv2
import numpy as np

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
# create a VideoCapture object to read the input video
#cap = cv2.VideoCapture(r"C:\Users\issac\Documents\badminton1.mp4")
cap = cv2.VideoCapture(r"C:\Users\issac\Documents\badminton4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
saliency = None
count = 0
name = "output3"
# create a motion saliency object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name+'.mp4',fourcc,fps,(1920, 1080),0)

while True:
    # read a frame from the input video
    
 
    ret, frame = cap.read()
    if not ret:
        break
    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        print(frame.shape[1],',',frame.shape[0])
        saliency.init()
        
    print(count)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    out.write(saliencyMap)  
    count+=1 
print("First done")
# release the VideoCapture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()



model = YOLO("Yolov8/SBadmintonbest.pt")
count =0

results = model.predict(source=r"C:\Users\issac\Documents\ML\output3.mp4", save =False, save_crop = False) # Display preds. Accepts all YOLO predict arguments
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('crop_mask3'+'.mp4',fourcc,30.0,(1920, 1080))

for result in results:
    mask = np.ones(result.orig_img.shape,dtype= result.orig_img.dtype)
    mask[:,:] = [255,255,255] 
    print(mask.shape)
    
    for bbox in result.boxes.xyxy:
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        crop_image=result.orig_img[int(y1):int(y2),int(x1):int(x2)]
        print("crop image shape=",crop_image.shape)
       
        mask[int(y1):int(y2),int(x1):int(x2)] = crop_image
    print("mask shape after=",mask.shape)
    
    count+=1
    print("count=",count)
    cv2.imwrite('C:/Users/issac/Documents/ML/Yolov8/SBadminton_mask/frame_'+str(count)+'.jpg', mask)
    out.write(mask)
out.release()

