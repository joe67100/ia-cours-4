from abc import ABC, abstractmethod


class DatasetHandler(ABC):
    @abstractmethod
    def download_dataset(self):
        pass

    @abstractmethod
    def export_annotations(self):
        pass
