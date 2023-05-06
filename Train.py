from ultralytics import YOLO
import wandb
# from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
if __name__ == '__main__': 
    model = YOLO("Yolov8/Yolo_models/yolov8n.pt")  # load a pretrained model (recommended for training)
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Badminton-Yolov8",)
# Use the model
    model.train(data=r"C:\Users\issac\Documents\ML\Yolov8\Only_Badminton\data.yaml", epochs=100)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
#yolo detect train data='C:\Users\issac\Documents\ML\Yolov8\shuttlecock\data.yaml' model=../yolov8n.pt epochs=100 imgsz=640

