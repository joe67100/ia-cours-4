import argparse
import os
import time

from dotenv import load_dotenv
from picsellia import Client

from src.Inference import Inference
from src.LocalFileHandler import LocalFileHandler
from src.PicselliaHandler import PicselliaHandler
from src.TrainingMediator import TrainingMediator
from src.YOLOTrainer import YOLOTrainer

load_dotenv()


def train(dataset_version: str, project_name: str) -> None:
    const = {
        "API_TOKEN": os.getenv("API_TOKEN"),
        "ORGANIZATION_NAME": os.getenv("ORGANIZATION_NAME"),
        "ANNOTATION_OUTPUT_PATH": os.getenv("ANNOTATION_OUTPUT_PATH"),
        "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME"),
        "DATASET_VERSION": dataset_version,
        "PROJECT_NAME": project_name,
    }

    client = Client(
        api_token=const["API_TOKEN"], organization_name=const["ORGANIZATION_NAME"]
    )
    picsellia_handler = PicselliaHandler(client)
    file_handler = LocalFileHandler()
    mediator = TrainingMediator(picsellia_handler, file_handler)

    mediator.prepare_data()

    experiment_name = f"experiment_{int(time.time())}"
    experiment = picsellia_handler.create_experiment(experiment_name)
    picsellia_handler.attach_dataset_to_experiment(experiment, const["DATASET_VERSION"])

    output_dir = "./datasets/structured"
    split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    yolo_trainer = YOLOTrainer(
        mediator, const["ANNOTATION_OUTPUT_PATH"], output_dir, split_ratios
    )
    yolo_trainer.prepare_directories()
    class_names = yolo_trainer.load_class_names()
    paired_files = yolo_trainer.pair_files("./datasets")
    splits = yolo_trainer.split_data(paired_files)
    yolo_trainer.move_files(splits, "./datasets")
    config_path = yolo_trainer.generate_config_yaml(class_names)
    yolo_trainer.train_yolo_model(config_path, experiment)

    model_version = picsellia_handler.attach_model_to_experiment(experiment)

    best_pt_path = file_handler.find_file("./runs", ".pt", "best")
    last_pt_path = file_handler.find_file("./runs", ".pt", "last")
    picsellia_handler.attach_files_to_model("best_pt", model_version, best_pt_path)
    picsellia_handler.attach_files_to_model("last_pt", model_version, last_pt_path)


def inference(mode: str, model: str, model_version: str, file_path: str | None) -> None:
    const = {
        "API_TOKEN": os.getenv("API_TOKEN"),
        "ORGANIZATION_NAME": os.getenv("ORGANIZATION_NAME"),
    }

    client = Client(
        api_token=const["API_TOKEN"], organization_name=const["ORGANIZATION_NAME"]
    )

    inference_instance = Inference(client, mode, model, model_version, file_path)
    inference_instance.infer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Training and Inference pipeline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--dataset_version", required=True, help="Dataset version"
    )
    train_parser.add_argument("--project_name", required=True, help="Project name")

    infer_parser = subparsers.add_parser("infer", help="Run inference pipeline")
    infer_parser.add_argument(
        "mode", choices=["video", "image", "camera"], help="Inference mode"
    )
    infer_parser.add_argument("--model", required=True, help="Model name")
    infer_parser.add_argument("--model_version", required=True, help="Model version")
    infer_parser.add_argument(
        "file_path",
        nargs="?",
        help="File path to a video/image (only required for video and image mode)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.dataset_version, args.project_name)
    elif args.command == "infer":
        if args.mode in ["video", "image"] and not args.file_path:
            parser.error("File path is required when mode is 'video' or 'image'")
        inference(args.mode, args.model, args.model_version, args.file_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
