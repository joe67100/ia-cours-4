from abc import ABC, abstractmethod


class DatasetHandler(ABC):
    @abstractmethod
    def download_dataset(self):
        pass

    @abstractmethod
    def export_annotations(self):
        pass

    @abstractmethod
    def create_experiment(self, experiment_name):
        pass

    @abstractmethod
    def attach_dataset_to_experiment(self, experiment, dataset_version_name):
        pass
