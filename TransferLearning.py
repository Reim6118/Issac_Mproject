# This code assumes that you have a dataset of 5 classes, stored in path/to/train/images, 
# and a validation set stored in path/to/val/images and path/to/val/annotations.json. 
# You can modify the code to suit your own dataset and needs.

import torch.nn as nn

from ultralytics import YOLO

# # Define the hyperparameters
# batch_size = 8
# num_workers = 4
# lr = 1e-3
# num_epochs = 10
# num_classes = 5

# Load the pre-trained YOLOv5 model
model = YOLO("yolov8n.pt")

# Replace the last layer with a new layer for your dataset
model.model[-1].model[-1] = nn.Conv2d(256, model.nc, kernel_size=(1, 1), stride=(1, 1))

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False
for param in model.model[-1].model[-1].parameters():
    param.requires_grad = True

model.train(data = "shuttlecock\data.yaml", epochs = 100)
model.val()
# # Load the new dataset
# train_transforms = transforms.Compose([
#     transforms.Resize((640, 640)),
#     transforms.ToTensor(),
# ])
# train_dataset = LoadImagesAndLabels('path/to/train/images', batch_size=batch_size, img_size=640, augment=True, transform=train_transforms)
# val_dataset = CocoDetection('path/to/val/images', 'path/to/val/annotations.json', transform=train_transforms)

# # Use DataLoader to load the data
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define the optimizer and loss function
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# criterion = nn.CrossEntropyLoss()

# # Train the new layer
# for epoch in range(num_epochs):
#     model.train()
#     for batch_i, (imgs, targets, paths, _) in enumerate(train_loader):
#         imgs = imgs.cuda().float()
#         targets = targets.cuda()
#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_i+1}/{len(train_loader)}], Loss: {loss.item()}")

# # Evaluate the model
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for imgs, targets in val_loader:
#         imgs = imgs.cuda().float()
#         targets = targets.cuda()
#         outputs = model(imgs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()
# print(f"Accuracy on validation set: {(correct/total)*100}%")
