from ultralytics import YOLO
from ultralytics.yolo.utils.benchmarks import benchmark
import cv2


class Model:
    """
    A class to build, train and evaluate YOLOv8
    """

    def __init__(self):
        self.model = None
        self.hyper_parameters = {}
        self.validation_results = None
        self.COLORS = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}

    def build(self, pretrained, model=None):
        """
        Creates the model either from scratch or load a pre-trained model.
        Available models are (from smallest to largest):
            * `YOLOv8n`
            * `YOLOv8s`
            * `YOLOv8m`
            * `YOLOv8l`
            * `YOLOv8x`

        See : https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models

        :param pretrained: load a pre-trained model or build a new model from scratch.
        :param model: If pretrained is True. Specify the model to load.
        """

        if pretrained:
            if model is None:
                raise ValueError('Model cannot be `None` when `pretrained` is True')
            else:
                self.model = YOLO(f"{model}.pt")

        # if `pretrained` is False, build from scratch
        else:
            self.model = YOLO("yolov8n.yaml")

    def load(self, path):
        """
        Load a model's weights.
        This is useful to load for example the last trained model directly
        instead of retraining it again.
        E.g.: path='runs/detect/train18/weights/best.pt'

        :param path:
        :return:
        """
        self.model = YOLO(path)

    def fit(self, data, epochs=10, patience=3, batch_size=16, img_size=640, save=True, optimizer='SGD', verbose=False,
            seed=123, resume=False, lr0=0.01, lrf=0.01, dropout=0.0):
        """
        Train the model on the `data`.

        :param data: the path to the `data.yaml` file. E.g.: `datasets/Guns-3/data.yaml`.
        :param epochs: the number of epochs.
        :param patience: number of epochs to wait before stopping if there is no improvement.
        :param: batch_size: number of images per batch. -1 for autobatch
        :param img_size: image size
        :param save: save train checkpoints
        :param optimzer: 'SGD', 'Adam', 'AdamW', 'RMSProp'
        :param verbose:
        :param seed: a random value for reproducibility
        :param resume: resume training from last checkpoint.
        :param lr0: initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        :param lrf: final learning rate (lr0 * lrf)
        :param dropout: use dropout regularization (classify train only)
        """

        self.hyper_parameters.update({
            'epochs': epochs,
            'batch size': batch_size,
            'image size': imgsz,
            'optimizer': optimizer,
            'learning rate initial': lr0,
            'learning rate final': lrf,
            'dropout': dropout
        })

        self.model.train(data=data, epochs=epochs, patience=patience, batch=batch_size, imgsz=img_size, save=save,
                         optimizer=optimizer, verbose=verbose, seed=seed, resume=resume, lr0=lr0, lrf=lrf,
                         dropout=dropout)

    def evaluate(self, split='val', batch=16, conf=0.001, save_hybrid=True, save_json=True, iou=0.6,
                 max_detect=30, imgsz=640, plots=False):
        """
        Evaluate the model on the validation data.
        Results are saved in `self.validation_results`.

        :param split: dataset split to use for validation. 'val', 'test' or 'train'
        :parm: batch: number of images per batch. -1 is for autobatch, default is 16
        :parm conf:	object confidence threshold for detection
        :param: save_hybrid: save hybrid version of labels (labels + additional predictions)
        :param: save_json: save results to JSON file
        :param: iou: intersection over union threshold for Non Maximum Suppression
        :param: max_det: maximum number of detections per image. 300 by default.
        :param: imgsz: image size. 640 by default.
        :param: plots: if True, show plots during training.
        """

        self.hyper_parameters.update({'validation_confidence_threshold': conf})

        self.validation_results = self.model.val(split=split, batch=batch, conf=conf, save_hybrid=save_hybrid,
                                                 save_json=save_json, iou=iou, max_det=max_detect)

    def predict(self, source, conf=0.25, stream=False, save=True, save_txt=True, save_conf=True, line_thickness=3):
        """
        Make inferences on the data source, then return the results in an array that contains
        informations such as bounding boxes coordinates and confidence, original image, etc.

        Source can be:
            * An image: PIL, OpenCV, etc.
            * URL: an url to an image or video.
            * a video
            * path: a path to an image or a directory.

        For large files, use `stream = True` to avoid filling up the memory.

        See: https://docs.ultralytics.com/modes/predict/.

        :param source: the source.
        :param conf: object confidence threshold for detection
        :param stream: if True, use stream mode.
        :param save: if True, save images with results
        :param save_txt: if True, save results as .txt file
        :param save_conf: if True,	save results with confidence scores.
        :param line_thickness: bounding box thickness

        :returns results: the results of the prediction.
        """

        self.hyper_parameters.update({'prediction_confidence_threshold': conf})

        results = self.model.predict(source=source, conf=conf, stream=stream, save=True, save_txt=save_txt,
                                     save_conf=save_conf, line_thickness=line_thickness)

        return results

    def predict_image(self, image, color=None):
        """
        Show the image with the predicted boxes.

        :param color: the color used to draw the predictions
        :param image: the image used to make inference.
        :return: pred_image: an image with predicted bounding boxes.
        """

        if color in self.COLORS.keys():
            color = self.COLORS.get(color)
        else:
            color = self.COLORS.get('red')

        pred_image = pred_results[0].plot()
        # pred_results = self.predict(source=image)
        # pred_image = self.draw_predicted_boxes(pred_results, color=color)

        return pred_image

    def predict_video(self, video_path, title='Predicted Video', color=None):
        """
        TODO: Save the video just like you do with the images

        Make inference on a video and draw predicted bounding boxes.

        :param video_path: a video
        :param title: the title of the frame used to show the video
        """

        if color in self.COLORS.keys():
            color = self.COLORS.get(color)
        else:
            color = self.COLORS.get('red')

        cap = cv2.VideoCapture(video_path)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame

                results = self.model.predict(source=frame)

                # get the predicted image with annotations
                # annotated_frame = results[0].plot()
                annotated_frame = self.draw_predicted_boxes(results)

                # Display the annotated frame
                cv2.imshow(title, annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def draw_predicted_boxes(self, pred_results, color=None):
        """
        Draw the predicted bounding boxes on the image.
        Note that predicted boxes can be retrieved with `pred_results[i].plot()`.

        :param pred_results: the results of the prediction. They are returned by `modelpredict()`
        :return: predicted_image: the image with predicted boxes.
        """
        if color in self.COLORS.keys():
            color = self.COLORS.get(color)
        else:
            color = self.COLORS.get('red')

        predicted_image = None

        # draw the predicted bounding boxes in the image
        for pred in pred_results:
            # get bounding box coordinates

            # check if gun coordinates are present
            if pred.boxes.xywh.tolist():
                bbox = pred.boxes.xywh.tolist()[0]
                bbox = np.array(bbox, dtype=int)

                original_image = pred.orig_img

                # x and y are the center of the box, so we put the at the top left
                w = bbox[2]
                h = bbox[3]
                x = int(bbox[0] - w / 2)
                y = int(bbox[1] - h / 2)

                # draw the bounding box
                predicted_image = cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)

                # draw class name and confidence
                class_name = pred.names.get(0)
                confidence = np.array(pred.boxes.conf)[0]
                confidence = np.round(confidence * 100, 2)
                text = class_name + ': ' + str(confidence)

                cv2.putText(img=predicted_image, text=text, org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=color, thickness=1, lineType=cv2.LINE_AA)

        return predicted_image

    def show_image_with_cv2(self, image, title='Predicted image'):
        """
        Show an image with cv2.

        :param image: image to show.
        :param title: the title of the image
        :return:
        """

        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def export(self, format):
        """
        Export the model to the specifed `format`.

        * `onnx`: a widely supported fromat. Use it for fast inference on web browsers.
        * `tfjs`: the Tensorflow JS format. Use it to train a model on web browsers.
        * `saved_model`: the Tensorfow format.

        See all format here: https://docs.ultralytics.com/modes/val/#arguments

        :param format: the format in which to export the model
        """

        self.model.export(format=format)

    def benchmark(self):
        """
        TODO:
        This allows us to find the optimal export format
        by providing metrics on various export formats for YOLOv8.

        It provides:
            * speed and accuracy
            * size
            * mAP50-95 (object detection) or accuracy_top5  (classification)
            * inference time in milliseconds per image

        :return:
        """
        self.benchmark = benchmark(model=self.model)

    def track(self, data):
        """
        See: https://docs.ultralytics.com/modes/track/#available-trackers

        :param data: a video
        :return:
        """

        raise NotImplementedError("Tracking has not been implemented yet.")

