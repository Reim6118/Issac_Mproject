from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("B_train5best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data=r"C:\Users\issac\Documents\ML\Yolov8\racket_shuttle\data.yaml", epochs=5)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
#yolo detect train data='C:\Users\issac\Documents\ML\Yolov8\shuttlecock\data.yaml' model=../yolov8n.pt epochs=100 imgsz=640