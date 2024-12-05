import os
from picsellia import Client
from dotenv import load_dotenv
from src.PicselliaDatasetHandler import PicselliaDatasetHandler
from src.LocalFileHandler import LocalFileHandler
from src.TrainingMediator import TrainingMediator
from src.YOLOTrainer import YOLOTrainer

load_dotenv()


def main():

    const = {
        "API_TOKEN": os.getenv("API_TOKEN"),
        "ORGANIZATION_NAME": os.getenv("ORGANIZATION_NAME"),
        "ANNOTATION_OUTPUT_PATH": os.getenv("ANNOTATION_OUTPUT_PATH"),
    }

    # Initialisation du client Picsellia
    client = Client(
        api_token=const["API_TOKEN"], organization_name=const["ORGANIZATION_NAME"]
    )
    dataset_handler = PicselliaDatasetHandler(client)
    file_handler = LocalFileHandler()
    mediator = TrainingMediator(dataset_handler, file_handler)

    mediator.prepare_data()

    annotations_dir = os.getenv("ANNOTATION_OUTPUT_PATH")
    output_dir = "./datasets/structured"
    split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    yolo_trainer = YOLOTrainer(mediator, annotations_dir, output_dir, split_ratios)
    yolo_trainer.prepare_directories()
    class_names = yolo_trainer.load_class_names()
    paired_files = yolo_trainer.pair_files("./datasets")
    splits = yolo_trainer.split_data(paired_files)
    yolo_trainer.move_files(splits, "./datasets")
    config_path = yolo_trainer.generate_config_yaml(class_names)
    yolo_trainer.train_yolo_model(config_path)


if __name__ == "__main__":
    main()
