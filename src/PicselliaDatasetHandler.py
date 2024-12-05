from .Abstract.DatasetHandler import DatasetHandler
from picsellia.types.enums import AnnotationFileType
import os
from dotenv import load_dotenv

load_dotenv()


class PicselliaDatasetHandler(DatasetHandler):
    def __init__(self, client):
        self.client = client
        self.dataset_name = os.getenv("DATASET_NAME")
        self.dataset_version = os.getenv("DATASET_VERSION")
        self.annotation_output_path = os.getenv("ANNOTATION_OUTPUT_PATH")
        self.dataset = self.client.get_dataset(self.dataset_name).get_version(
            self.dataset_version
        )

    def download_dataset(self):
        if not os.path.exists("./datasets"):
            self.dataset.list_assets().download("./datasets")

    def export_annotations(self):
        self.dataset.export_annotation_file(
            AnnotationFileType.YOLO, self.annotation_output_path
        )
