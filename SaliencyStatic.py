import cv2
import glob

image_filenames = glob.glob(r"C:\Users\issac\Documents\ML\Yolov8\Ball\test\images\*.jpg")
saliency = None
for filename in image_filenames:
    # read the image
    img = cv2.imread(filename)
    # if saliency is None:
    #     saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
    #     saliency.setImagesize(img.shape[1], img.shape[0])
    #     saliency.init()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (success, saliencyMap) = saliency.computeSaliency(gray)
    # saliencyMap = (saliencyMap * 255).astype("uint8")

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # show the images
    cv2.imshow("Image", img)
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
        break