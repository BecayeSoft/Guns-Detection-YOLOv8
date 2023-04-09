from comet_ml import Experiment
import os
from dotenv import load_dotenv


class Monitor:
    """
    This class acts as an interface to interact with Comet.
    It contanis methods to log the hyperparameters, performances metrics and upload the model to Comet.

    See: https://www.comet.com/becayesoft/guns-detection
    """

    def __init__(self, project_name="Guns-Detection", workspace="becayesoft"):
        load_dotenv('credentials.env')
        key = os.getenv('COMET_API_KEY')

        self.experiment = Experiment(
            api_key=key,
            project_name=project_name,
            workspace=workspace,
            log_code=True
        )


    def log_hyper_parameters(self, hyper_params):
        """
        Log the training hyperparameters and the validation and inference confidences.
        :param train_param: a dictionary containing the hyperparameters use to train the model.
        :param val_confidence: confidence used to evaluate the model.
        :param pred_confidence: confidence used to make new predictions.
        """

        self.experiment.log_parameters(hyper_params)


    def log_performance_metrics(self, val_results):
        """
        Log the model's performances metrics and the speed.

        :param val_results: the results of the validation, returned by `model.evaluate()`.
        :return:
        """

        # Get, rename and round performance metrics
        performance_metrics = {
            "precision": val_results.results_dict['metrics/precision(B)'],
            "recall": val_results.results_dict['metrics/recall(B)'],
            "mAP50": val_results.results_dict['metrics/mAP50(B)'],
            "mAP50_95": val_results.results_dict['metrics/mAP50-95(B)'],
            "fitness": val_results.results_dict['fitness']
        }
        for key, value in performance_metrics.items():
            performance_metrics[key] = round(value, 3)

        # Add speed metrics
        performance_metrics.update(val_results.speed)

        # log to comet
        self.experiment.log_metrics(performance_metrics, step=None, epoch=None)


    def upload_model(self, path_to_model, name='YOLOv8'):
        """
        Upload the model to Comet.

        :param path_to_model: the path to the model. E.g: 'runs/detect/train18/weights/best.pt'
        """

        self.experiment.log_model(name=name, file_or_folder=path)


    def end_experiment():
        """
        End the experiment.
        """

        self.experiment.end()

