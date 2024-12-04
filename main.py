import os
import random
import shutil
import zipfile
import yaml
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO
from dotenv import load_dotenv

def train_model():
    load_dotenv()
    cwd = os.getcwd()

    const = {
        "API_TOKEN": os.getenv("API_TOKEN"),
        "ORGANIZATION_NAME": os.getenv("ORGANIZATION_NAME"),
        "DATASET_NAME": os.getenv("DATASET_NAME"),
        "DATASET_VERSION": os.getenv("DATASET_VERSION"),
        "PROJECT_NAME": os.getenv("PROJECT_NAME"),
        "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME"),
    }

    # Initialisation du client Picsellia
    client = Client(api_token=const["API_TOKEN"], organization_name=const["ORGANIZATION_NAME"])
    dataset = client.get_dataset(const["DATASET_NAME"]).get_version(const["DATASET_VERSION"])

    # Télécharger le dataset si le dossier n'existe pas
    if not os.path.exists("./datasets"):
        dataset.list_assets().download("./datasets")

    # Initialisation du projet et création de l'expérience
    project = client.get_project(const["PROJECT_NAME"])
    experiment = project.get_experiment(const["EXPERIMENT_NAME"])

    # Exporter les annotations au format YOLO
    annotation_output_path = "./datasets/annotations"
    dataset.export_annotation_file(AnnotationFileType.YOLO, annotation_output_path)

    # Fonction pour trouver le fichier ZIP
    def find_file(directory, extension):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    return os.path.join(root, file)
        return None

    # Trouver et extraire le fichier ZIP
    zip_file_path = find_file(annotation_output_path, ".zip")
    if zip_file_path:
        annotations_dir = os.path.join("./datasets", "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(annotations_dir)

        os.remove(zip_file_path)
    else:
        print("Aucun fichier ZIP trouvé dans les annotations.")

    # Trouver le fichier YAML et charger les classes
    data_yaml_path = find_file(annotations_dir, ".yaml")
    if data_yaml_path:
        with open(data_yaml_path, "r") as yaml_file:
            class_names = yaml.safe_load(yaml_file).get("names", [])
    else:
        print("Fichier data.yml introuvable.")
        exit(1)

    # Structurer les données pour YOLO
    output_dir = "./datasets/structured"
    images_dir = f"{output_dir}/images"
    labels_dir = f"{output_dir}/labels"
    split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    # Créer les répertoires de sortie
    for dir in [images_dir, labels_dir]:
        for split in split_ratios:
            os.makedirs(os.path.join(dir, split), exist_ok=True)

    # Liste des fichiers d'images et annotations
    base_dir = "./datasets"
    image_files = [f for f in os.listdir(base_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    label_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

    image_to_label = {
        img: img.replace(img.split('.')[-1], "txt") for img in image_files
    }
    paired_files = [(img, lbl) for img, lbl in image_to_label.items() if lbl in label_files]

    # Répartition des données en train/val/test
    random.shuffle(paired_files)
    n_total = len(paired_files)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])

    splits = {
        "train": paired_files[:n_train],
        "val": paired_files[n_train:n_train + n_val],
        "test": paired_files[n_train + n_val:],
    }

    # Déplacer les fichiers vers les répertoires correspondants
    for split, files in splits.items():
        for img, lbl in files:
            img_path = os.path.join(base_dir, img)
            lbl_path = os.path.join(annotations_dir, lbl)
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                shutil.move(img_path, os.path.join(images_dir, split, img))
                shutil.move(lbl_path, os.path.join(labels_dir, split, lbl))
            else:
                print(f"Fichier manquant pour {img} ou {lbl}.")

    # Génération du fichier config.yaml pour YOLO
    config = {
        "path": os.path.join(cwd, "datasets/structured"),
        "train": os.path.join(cwd, "datasets/structured/images/train"),
        "val": os.path.join(cwd, "datasets/structured/images/val"),
        "test": os.path.join(cwd, "datasets/structured/images/test"),
        "nc": len(class_names),
        "names": class_names,
    }

    # Sauvegarder le fichier config.yaml
    config_path = "./config.yaml"
    with open(config_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    # Charger et entraîner le modèle YOLO
    model = YOLO("yolo11n.pt")
    hyperparameters = {
        "epochs": 100,
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
        "patience": 10,
    }

    # Entraînement du modèle
    model.train(data="./config.yaml", **hyperparameters)

if __name__ == "__main__":
    train_model()
