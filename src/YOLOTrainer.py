import os
import random
import shutil
import yaml
import torch
from ultralytics import YOLO
from picsellia.types.enums import LogType, AddEvaluationType, InferenceType
from picsellia.sdk.experiment import Experiment
from src.PicselliaLogger import PicselliaLogger
from src.TrainingMediator import TrainingMediator


class YOLOTrainer:
    """
    A class to handle the training process of a YOLO model.

    Attributes:
        mediator (TrainingMediator): An instance of TrainingMediator to handle data preparation.
        annotations_dir (str): Directory containing annotation files.
        output_dir (str): Directory to store output files.
        split_ratios (dict[str, float]): Ratios for splitting the dataset into train, validation, and test sets.
    """

    def __init__(
        self,
        mediator: TrainingMediator,
        annotations_dir: str,
        output_dir: str,
        split_ratios: dict[str, float],
    ) -> None:
        self.mediator = mediator
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.images_dir = f"{output_dir}/images"
        self.labels_dir = f"{output_dir}/labels"

    def prepare_directories(self) -> None:
        """
        Prepares the necessary directories for storing images and labels.

        Creates directories for train, validation, and test splits within the images and labels directories.
        """
        for dir in [self.images_dir, self.labels_dir]:
            for split in self.split_ratios:
                os.makedirs(os.path.join(dir, split), exist_ok=True)

    def load_class_names(self) -> list:
        """
        Loads class names from a YAML file.

        Returns:
            list: A list of class names.

        Raises:
            SystemExit: If the YAML file is not found.
        """
        data_yaml_path = self.mediator.file_handler.find_file(
            self.annotations_dir, ".yaml"
        )
        if data_yaml_path:
            with open(data_yaml_path, "r") as yaml_file:
                return yaml.safe_load(yaml_file).get("names", [])
        else:
            print("Fichier data.yml introuvable.")
            exit(1)

    def pair_files(self, base_dir: str) -> list:
        """
        Pairs image files with their corresponding label files.

        Args:
            base_dir (str): The base directory containing image files.

        Returns:
            list: A list of tuples, each containing an image file and its corresponding label file.
        """
        image_files = [
            f for f in os.listdir(base_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        label_files = [
            f for f in os.listdir(self.annotations_dir) if f.endswith(".txt")
        ]
        image_to_label = {
            img: img.replace(img.split(".")[-1], "txt") for img in image_files
        }
        return [(img, lbl) for img, lbl in image_to_label.items() if lbl in label_files]

    def split_data(self, paired_files: list) -> dict:
        """
        Splits the paired files into train, validation, and test sets.

        Args:
            paired_files (list): A list of paired image and label files.

        Returns:
            dict: A dictionary containing the train, validation, and test splits.
        """
        random.shuffle(paired_files)
        n_total = len(paired_files)
        n_train = int(n_total * self.split_ratios["train"])
        n_val = int(n_total * self.split_ratios["val"])
        return {
            "train": paired_files[:n_train],
            "val": paired_files[n_train : n_train + n_val],
            "test": paired_files[n_train + n_val :],
        }

    def move_files(self, splits: dict, base_dir: str) -> None:
        """
        Moves files to their respective directories based on the split.

        Args:
            splits (dict): A dictionary containing the train, validation, and test splits.
            base_dir (str): The base directory containing the files to be moved.
        """
        for split, files in splits.items():
            for img, lbl in files:
                img_path = os.path.join(base_dir, img)
                lbl_path = os.path.join(self.annotations_dir, lbl)
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    shutil.move(img_path, os.path.join(self.images_dir, split, img))
                    shutil.move(lbl_path, os.path.join(self.labels_dir, split, lbl))
                else:
                    print(f"Fichier manquant pour {img} ou {lbl}.")

    def generate_config_yaml(self, class_names: list) -> str:
        """
        Generates a configuration YAML file for the YOLO model.

        Args:
            class_names (list): A list of class names.

        Returns:
            str: The path to the generated configuration YAML file.
        """
        config = {
            "path": os.path.join(os.getcwd(), "datasets/structured"),
            "train": os.path.join(os.getcwd(), "datasets/structured/images/train"),
            "val": os.path.join(os.getcwd(), "datasets/structured/images/val"),
            "test": os.path.join(os.getcwd(), "datasets/structured/images/test"),
            "nc": len(class_names),
            "names": class_names,
        }
        config_path = "./config.yaml"
        with open(config_path, "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        return config_path

    def initialize_model(self) -> YOLO:
        """
        Initializes the YOLO model.

        Returns:
            YOLO: An instance of the YOLO model.
        """
        model = YOLO("yolo11n.pt")
        if torch.cuda.is_available():
            model.to("cuda")
        elif torch.backends.mps.is_available():
            model.to("mps")
        else:
            print("CUDA / MPS not available. Using CPU.")
        return model

    def set_hyperparameters(self) -> dict:
        return {
            "epochs": 500,
            "batch": 8,
            "imgsz": 512,
            "close_mosaic": 0,
            "optimizer": "AdamW",
            "lr0": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0004,
            "seed": 42,
            "augment": True,
            "cache": True,
            "label_smoothing": 0.05,
            "mosaic": True,
            "mixup": True,
            "patience": 100,
        }

    def add_callbacks(self, model: YOLO, experiment: Experiment) -> None:
        """
        Adds callbacks to the YOLO model for logging stuff on Picsellia.

        Args:
            model (YOLO): An instance of the YOLO model.
            experiment (Experiment): An instance of the Experiment class for logging.
        """
        picsellia_logger = PicselliaLogger(experiment)
        model.add_callback("on_train_epoch_end", picsellia_logger.on_train_epoch_end)
        model.add_callback("on_train_end", picsellia_logger.on_train_end)

    def train_model(self, model: YOLO, config_path: str, hyperparameters: dict) -> None:
        model.train(data=config_path, **hyperparameters)

    def evaluate_model(self, model: YOLO, experiment: Experiment) -> None:
        """
        Evaluates the YOLO model and logs some elements on Picsellia.

        Args:
            model (YOLO): An instance of the YOLO model.
            experiment (Experiment): An instance of the Experiment class for logging.
        """
        results = model.val(data="config.yaml")
        experiment.log("best fitness", float(results.fitness), LogType.VALUE)
        for key, value in results.results_dict.items():
            if "precision" in key or "recall" in key:
                experiment.log(f"overall {key} value", float(value), LogType.VALUE)

    def add_evaluation_to_picsellia(self, model: YOLO, experiment: Experiment) -> None:
        """
        Uses the trained model to select some images from the validation dataset,
        makes predictions, and adds evaluations to Picsellia.

        Args:
            model (YOLO): An instance of the YOLO model.
            experiment (Experiment): An instance of the Experiment class for logging.
        """
        with open("config.yaml", "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            names = config["names"]
            print(names)

        val_images_dir = os.path.join(self.images_dir, "val")
        val_images = [
            img
            for img in os.listdir(val_images_dir)
            if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        selected_images = random.sample(val_images, min(30, len(val_images)))

        dataset_version = experiment.get_dataset("initial")
        dataset_labels = {label.name: label for label in dataset_version.list_labels()}

        for img in selected_images:
            image_path = os.path.join(val_images_dir, img)
            img_id = os.path.splitext(img)[0]  # Remove the file extension

            try:
                asset = dataset_version.find_asset(id=img_id)
                results = model(image_path)
                for result in results:
                    rectangles = []
                    classifications = []

                    for box in result.boxes:
                        x_center, y_center, w, h = map(float, box.xywh[0])

                        # Used to convert center-center values to top-left values (used in Picsellia)
                        x = int(x_center - w / 2)
                        y = int(y_center - h / 2)

                        label_id = int(box.cls[0].item())
                        confidence = box.conf[0].item()
                        label_name = names[label_id]  # Get label name from names list
                        if label_name in dataset_labels:
                            label = dataset_labels[label_name]
                            rectangles.append((x, y, int(w), int(h), label, confidence))
                            classifications.append((label, confidence))
                        else:
                            print(
                                f"Label name {label_name} not found in dataset labels."
                            )

                    experiment.add_evaluation(
                        asset,
                        rectangles=rectangles,
                        add_type=AddEvaluationType.REPLACE,
                        classifications=classifications,
                    )
            except Exception as e:
                print(f"An error occurred: {e}")

        job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
        job.wait_for_done()

    def train_yolo_model(self, config_path: str, experiment: Experiment) -> None:
        """
        Orchestrates the entire training process of the YOLO model.

        Args:
            config_path (str): The path to the configuration YAML file.
            experiment (Experiment): An instance of the Experiment class for logging.
        """
        model = self.initialize_model()
        hyperparameters = self.set_hyperparameters()
        experiment.log_parameters(hyperparameters)
        self.add_callbacks(model, experiment)
        self.train_model(model, config_path, hyperparameters)
        self.evaluate_model(model, experiment)
        self.add_evaluation_to_picsellia(model, experiment)
