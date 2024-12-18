from .Abstract.DatasetHandler import DatasetHandler
from picsellia.types.enums import AnnotationFileType
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia import Client
import os
from dotenv import load_dotenv

load_dotenv()


class PicselliaDatasetHandler(DatasetHandler):
    def __init__(self, client: Client):
        self.client = client
        self.dataset_name = os.getenv("DATASET_NAME")
        self.dataset_version = os.getenv("DATASET_VERSION")
        self.annotation_output_path = os.getenv("ANNOTATION_OUTPUT_PATH")
        self.dataset: DatasetVersion = self.client.get_dataset(
            self.dataset_name
        ).get_version(self.dataset_version)

    def download_dataset(self):
        if not os.path.exists("./datasets"):
            self.dataset.list_assets().download("./datasets", use_id=True)

    def export_annotations(self):
        self.dataset.export_annotation_file(
            AnnotationFileType.YOLO, self.annotation_output_path, use_id=True
        )
