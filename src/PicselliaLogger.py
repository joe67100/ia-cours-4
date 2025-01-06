class PicselliaLogger:
    def __init__(self, experiment):
        self.experiment = experiment

    def on_train_epoch_end(self, trainer):
        metrics = trainer.metrics
        for key, value in metrics.items():
            self.experiment.log(key, value, "LINE")
