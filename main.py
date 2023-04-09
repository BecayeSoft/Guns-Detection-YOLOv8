from ultralytics import YOLO
from DataFlow import DataFlow
from Model import Model
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# import shutil


# Load the data
flow = DataFlow(version=3)
dataset, data_yaml_path = flow.load_dataset()

# Load model's weights
model = Model()
model.load(path='runs/detect/train18/weights/best.pt')

# Build & Train the model
# model.build(pretrained=True, model='yolov8n')
# model.fit(data=data_yaml_path, epochs=15)

# Evaluate the model
model.evaluate()

# Predict new image
new_image = Image.open("assets/test.jpg")
model.predict_image(image=new_image)


# tensorboard
# !tensorboard --logdir runs/detect/train5

# Export the model
model.export(format='saved_model')
