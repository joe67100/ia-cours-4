import os

from dotenv import load_dotenv

load_dotenv()


class TrainingMediator:
    def __init__(self, dataset_handler, file_handler):
        self.dataset_handler = dataset_handler
        self.file_handler = file_handler

    def prepare_data(self):
        self.dataset_handler.download_dataset()
        self.dataset_handler.export_annotations()

        zip_file_path = self.file_handler.find_file(
            os.getenv("ANNOTATION_OUTPUT_PATH"), ".zip"
        )
        if zip_file_path:
            self.file_handler.extract_zip(
                zip_file_path, os.getenv("ANNOTATION_OUTPUT_PATH")
            )
        else:
            print("Aucun fichier ZIP trouvé dans les annotations.")
