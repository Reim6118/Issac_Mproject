
from ultralytics import YOLO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

model = YOLO("Yolov8/SBadmintonbest.pt")
count =0
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
results = model.predict(source=r"C:\Users\issac\Documents\ML\output1.mp4", save =False, save_crop = False) # Display preds. Accepts all YOLO predict arguments
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('crop_mask4'+'.mp4',fourcc,30.0,(1920, 1080))
out2 = cv2.VideoWriter('crop_original'+'.mp4',fourcc,30.0,(1920, 1080))
cap = cv2.VideoCapture(r"C:\Users\issac\Documents\badminton2.mp4")

# for result in results:
#     cv2.imshow('result',result.masks)
#     if cv2.waitKey(1) == ord('q'):
#         break   
# cv2.destroyAllWindows()

# for result in results:
#     mask = np.ones(result.orig_img.shape, dtype= result.orig_img.dtype)
#     mask[:,:] = [255,255,255] 
#     cv2.imshow('mask',mask)
#     print(mask.shape)
#     out.write(mask)
# out.release()
for result in results:
    ret, frame = cap.read()
    # mask = np.ones(result.orig_img.shape,dtype= result.orig_img.dtype)
    mask2 = np.ones(frame.shape,dtype= frame.dtype)

    # mask[:,:] = [255,255,255]
    mask2[:,:] = [255,255,255] 
    # print(mask.shape)
    
    for bbox in result.boxes.xyxy:
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        # crop_image=result.orig_img[int(y1):int(y2),int(x1):int(x2)]
        crop_original=frame[int(y1):int(y2),int(x1):int(x2)]
        # print("crop image shape=",crop_image.shape)
        # rgb_crop = cv2.cvtColor(crop_image,cv2.COLOR_GRAY2RGB)
        # print("rgb crop image shape=",rgb_crop.shape)
        # mask[int(y1):int(y2),int(x1):int(x2)] = crop_image
        mask2[int(y1):int(y2),int(x1):int(x2)] = crop_original
    # print("mask shape after=",mask.shape)
    print("mask2 shape",mask2.shape)
    
    # count+=1
    # print("count=",count)
    # cv2.imwrite('C:/Users/issac/Documents/ML/Yolov8/SBadminton_mask/frame_'+str(count)+'.jpg', mask)
    # out.write(mask)
    out2.write(mask2)
# out.release()
out2.release()


# # from PIL
# im1 = Image.open(r"C:\Users\issac\Documents\badminton.mp4")
# results = model.predict(source=im1, save=True)  # save plotted images

# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])