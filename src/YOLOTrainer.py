import os
import random
import shutil
import yaml
from ultralytics import YOLO
from picsellia.sdk.experiment import Experiment
from src.PicselliaLogger import PicselliaLogger
from src.TrainingMediator import TrainingMediator


class YOLOTrainer:
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
        for dir in [self.images_dir, self.labels_dir]:
            for split in self.split_ratios:
                os.makedirs(os.path.join(dir, split), exist_ok=True)

    def load_class_names(self) -> list:
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
        model = YOLO("yolo11n.pt")
        model.to("cuda")
        return model

    def set_hyperparameters(self) -> dict:
        return {
            "epochs": 1,
            "batch": 32,
            "imgsz": 512,
            "close_mosaic": 0,
            "optimizer": "AdamW",
            "lr0": 0.0025,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "seed": 42,
            "augment": True,
            "cache": True,
            "label_smoothing": 0.1,
            "mosaic": True,
            "patience": 100,
        }

    def add_callbacks(self, model: YOLO, experiment: Experiment) -> None:
        picsellia_logger = PicselliaLogger(experiment)
        model.add_callback("on_train_epoch_end", picsellia_logger.on_train_epoch_end)
        model.add_callback("on_train_end", picsellia_logger.on_train_end)

    def train_model(self, model: YOLO, config_path: str, hyperparameters: dict) -> None:
        model.train(data=config_path, **hyperparameters)

    def evaluate_model(self, model: YOLO) -> None:
        results = model.val(data="config.yaml")
        print("Class indices with average precision:", results.ap_class_index)
        print("Average precision for all classes:", results.box.all_ap)
        print("Average precision:", results.box.ap)
        print("Average precision at IoU=0.50:", results.box.ap50)
        print("Class indices for average precision:", results.box.ap_class_index)
        print("Class-specific results:", results.box.class_result)
        print("F1 score:", results.box.f1)
        print("F1 score curve:", results.box.f1_curve)
        print("Overall fitness score:", results.box.fitness)
        print("Mean average precision:", results.box.map)
        print("Mean average precision at IoU=0.50:", results.box.map50)
        print("Mean average precision at IoU=0.75:", results.box.map75)
        print("Mean average precision for different IoU thresholds:", results.box.maps)
        print("Mean results for different metrics:", results.box.mean_results)
        print("Mean precision:", results.box.mp)
        print("Mean recall:", results.box.mr)
        print("Precision:", results.box.p)
        print("Precision curve:", results.box.p_curve)
        print("Precision values:", results.box.prec_values)
        print("Specific precision metrics:", results.box.px)
        print("Recall:", results.box.r)
        print("Recall curve:", results.box.r_curve)

    def train_yolo_model(self, config_path: str, experiment: Experiment) -> None:
        model = self.initialize_model()
        hyperparameters = self.set_hyperparameters()
        self.add_callbacks(model, experiment)
        self.train_model(model, config_path, hyperparameters)
        self.evaluate_model(model)
