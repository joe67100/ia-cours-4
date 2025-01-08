import os

from dotenv import load_dotenv

from src.PicselliaHandler import PicselliaHandler
from src.Abstract.FileHandler import FileHandler

load_dotenv()


class TrainingMediator:
    def __init__(self, picsellia_handler: PicselliaHandler, file_handler: FileHandler) -> None:
        self.picsellia_handler = picsellia_handler
        self.file_handler = file_handler

    def prepare_data(self) -> None:
        self.picsellia_handler.download_dataset()
        self.picsellia_handler.export_annotations()

        zip_file_path = self.file_handler.find_file(
            os.getenv("ANNOTATION_OUTPUT_PATH"), ".zip"
        )
        if zip_file_path:
            self.file_handler.extract_zip(
                zip_file_path, os.getenv("ANNOTATION_OUTPUT_PATH")
            )
        else:
            print("Aucun fichier ZIP trouvé dans les annotations.")
