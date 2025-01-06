import os
import random
import shutil
import yaml
from ultralytics import YOLO

from src.PicselliaLogger import PicselliaLogger


class YOLOTrainer:
    def __init__(self, mediator, annotations_dir, output_dir, split_ratios):
        self.mediator = mediator
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.images_dir = f"{output_dir}/images"
        self.labels_dir = f"{output_dir}/labels"

    def prepare_directories(self):
        for dir in [self.images_dir, self.labels_dir]:
            for split in self.split_ratios:
                os.makedirs(os.path.join(dir, split), exist_ok=True)

    def load_class_names(self):
        data_yaml_path = self.mediator.file_handler.find_file(
            self.annotations_dir, ".yaml"
        )
        if data_yaml_path:
            with open(data_yaml_path, "r") as yaml_file:
                return yaml.safe_load(yaml_file).get("names", [])
        else:
            print("Fichier data.yml introuvable.")
            exit(1)

    def pair_files(self, base_dir):
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

    def split_data(self, paired_files):
        random.shuffle(paired_files)
        n_total = len(paired_files)
        n_train = int(n_total * self.split_ratios["train"])
        n_val = int(n_total * self.split_ratios["val"])
        return {
            "train": paired_files[:n_train],
            "val": paired_files[n_train : n_train + n_val],
            "test": paired_files[n_train + n_val :],
        }

    def move_files(self, splits, base_dir):
        for split, files in splits.items():
            for img, lbl in files:
                img_path = os.path.join(base_dir, img)
                lbl_path = os.path.join(self.annotations_dir, lbl)
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    shutil.move(img_path, os.path.join(self.images_dir, split, img))
                    shutil.move(lbl_path, os.path.join(self.labels_dir, split, lbl))
                else:
                    print(f"Fichier manquant pour {img} ou {lbl}.")

    def generate_config_yaml(self, class_names):
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

    def train_yolo_model(self, config_path, experiment):
        model = YOLO("yolo11n.pt")
        model.to("cuda")
        hyperparameters = {
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
        picsellia_logger = PicselliaLogger(experiment)
        model.add_callback("on_train_epoch_end", picsellia_logger.on_train_epoch_end)
        ## TODO : Log on_val_epoch_end
        model.train(data=config_path, **hyperparameters)
