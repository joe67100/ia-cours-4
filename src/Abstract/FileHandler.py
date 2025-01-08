from abc import ABC, abstractmethod


class FileHandler(ABC):
    @abstractmethod
    def find_file(self, directory: str, extension: str) -> str | None:
        pass

    @abstractmethod
    def extract_zip(self, zip_file_path, extract_to) -> None:
        pass
