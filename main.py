from ultralytics import YOLO
from DataFlow import DataFlow
from Model import Model
from Monitor import Monitor


# Load the data
flow = DataFlow(version=3)
dataset, data_yaml_path = flow.load_dataset()

# ------------------------------------------------------
# Training
# Load the model's weights, evaluate it, export it
# ------------------------------------------------------

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
# model.predict_image(image=new_image)
model.predict_video(video_path="C:/Users/balde/OneDrive/Bureau/pexels-karolina-grabowska-5243141-1920x1080-50fps.mp4")

# Export the model
model.export(format='onnx')


# -------------------------------------------------------------
# Monitoring
# Log hyperparameters, performance metrics, and upload the model
# -------------------------------------------------------------
monitor = Monitor(project_name='Guns-Detecions-YOLOv8')
monitor.log_hyper_parameters(model.hyper_parameters)
monitor.log_performance_metrics(model.validation_results)
monitor.upload_model(path='runs/detect/train18/weights/best.pt', name='guns_model')
