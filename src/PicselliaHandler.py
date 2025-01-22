import os

from dotenv import load_dotenv
from picsellia import Client
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.model_version import ModelVersion
from picsellia.types.enums import AnnotationFileType

load_dotenv()


class PicselliaHandler:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.dataset_name = os.getenv("DATASET_NAME")
        self.dataset_version = os.getenv("DATASET_VERSION")
        self.annotation_output_path = os.getenv("ANNOTATION_OUTPUT_PATH")
        self.project_name = os.getenv("PROJECT_NAME")
        self.dataset: DatasetVersion = self.client.get_dataset(
            self.dataset_name
        ).get_version(self.dataset_version)

    def download_dataset(self) -> None:
        if not os.path.exists("./datasets"):
            self.dataset.list_assets().download("./datasets", use_id=True)

    def export_annotations(self) -> None:
        self.dataset.export_annotation_file(
            AnnotationFileType.YOLO, self.annotation_output_path, use_id=True
        )

    def create_experiment(self, experiment_name: str) -> Experiment:
        project = self.client.get_project(self.project_name)
        experiment = project.create_experiment(name=experiment_name)
        return experiment

    def attach_dataset_to_experiment(
        self, experiment: Experiment, dataset_version_name: str
    ) -> None:
        experiment.attach_dataset(
            name=dataset_version_name, dataset_version=self.dataset
        )

    def attach_model_to_experiment(self, experiment: Experiment) -> ModelVersion:
        existing_model = self.client.get_model("Groupe_2")
        model_version = experiment.export_in_existing_model(existing_model)
        print(f"Model version {model_version.name} created.")
        experiment.attach_model_version(model_version)
        return model_version

    def attach_files_to_model(
        self, file_name: str, model_version: ModelVersion, pt_path: str
    ) -> None:
        model_version.store(file_name, pt_path)
