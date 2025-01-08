import numpy as np
from picsellia.sdk.experiment import Experiment



class PicselliaLogger:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_train_epoch_end(self, trainer) -> None:
        metrics = trainer.metrics
        for key, value in metrics.items():
            self.experiment.log(key, value, "LINE")

    def on_train_end(self, trainer) -> None:
        metrics = trainer.metrics
        print("Metrics at training end:", metrics)
        print("Metrics object type:", type(trainer.metrics))

        for key, value in metrics.items():
            # Ensure compatibility with JSON logging systems
            if isinstance(value, (np.float64, np.float32)):
                value = float(value)
            print(f"Logging: {key} -> {value}")
            self.experiment.log(f"train_end_{key}", value, "LINE")
