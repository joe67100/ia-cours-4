import numpy as np
from picsellia.sdk.experiment import Experiment
from picsellia.exceptions import BadRequestError


class PicselliaLogger:
    """
    A class to handle logging of training metrics to Picsellia.

    Attributes:
        experiment (Experiment): An instance of the Experiment class for logging.
    """

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_train_epoch_end(self, trainer) -> None:
        metrics = trainer.metrics
        for key, value in metrics.items():
            try:
                self.experiment.log(key, value, "LINE")
            except BadRequestError as e:
                print(f"Failed to log {key}: {e}")

    def on_train_end(self, trainer) -> None:
        metrics = trainer.metrics

        for key, value in metrics.items():
            # Ensure compatibility with JSON logging systems
            if isinstance(value, (np.float64, np.float32)):
                value = float(value)
            print(f"Logging: {key} -> {value}")
            try:
                self.experiment.log(f"train_end_{key}", value, "LINE")
            except BadRequestError as e:
                print(f"Failed to log {key}: {e}")
