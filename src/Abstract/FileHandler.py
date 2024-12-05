from abc import ABC, abstractmethod


class FileHandler(ABC):
    @abstractmethod
    def find_file(self, directory, extension):
        pass

    @abstractmethod
    def extract_zip(self, zip_file_path, extract_to):
        pass
