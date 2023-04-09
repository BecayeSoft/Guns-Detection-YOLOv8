import os

os.environ["HOME"] = "C:/Users/balde"  # fix Roboflow issue
from roboflow import Roboflow
import shutil
from dotenv import load_dotenv


class DataFlow:
    # TODO Update class so it can load data from different workspaces and projets

    """
    A class that connects to Roboflow to load the dataset,
    load the YOLO8 model from roboflow, and make predictions.
    """

    def __init__(self, workspace="yolo-xkggu", project="guns-mms73", version=3):
        """
        Load the specified `project` from roboflow

        :param workspace: roboflow workspace
        :param project: project id
        """

        load_dotenv('credentials.env')
        key = os.getenv('ROBOFLOW_API_KEY')
        rf = Roboflow(key)

        self.project = rf.workspace(workspace).project(project)
        self.version = version

    def load_dataset(self):
        """
        Load the dataset from roboflow.
        then move it to the `datasets` folder  project loaded when creating the object.

        Note: the dataset is downloaded in the folder `name-version`. E.g: `Guns-3`.

        :return: `(dataset, data_yaml_path)` the roboflow dataset and the path to
        the data.yaml file which contains information for training the model.
        """

        dataset = self.project.version(self.version).download("yolov8")
        original_dir = f"{dataset.name}-{dataset.version}"
        dataset_dir = "datasets/"

        # create `datasets/` directory if it doesn't exist
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # move the dataset to `datasets/`
        try:
            shutil.move(original_dir, dataset_dir)
        except shutil.Error:
            print('Warning: Destination path already exists\nCheck if the dataset has not already been loaded.')

        data_yaml_path = os.path.join(dataset.location, 'data.yaml')

        return dataset, data_yaml_path

    def load_model(self):
        """
        Load the model from Roboflow
        """

        self.model = self.project.version(self.version).model

    def predict(self, data, confidence=40, overlap=30, save=True):
        """
        Predict the data.

        :param data:
        :param confidence:
        :param overlap:
        :param save:
        :return: the result of the prediction
        """

        result = self.model.predict(data, confidence=confidence, overlap=overlap).json()
        # print(model.predict("IMAGE_URL", hosted=True, confidence=40, overlap=30).json())

        if save is True:
            self.model.predict(data, confidence=confidence, overlap=overlap).save("prediction.jpg")

        return result
