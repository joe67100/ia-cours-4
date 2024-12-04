import os
import random
import shutil
import zipfile
import yaml
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO

# Paramètres fixes
API_TOKEN = "79c34d713ac3d85a03d1805855086c4bc00d1225"
ORGANIZATION_NAME = "Picsalex-MLOps"
PROJECT_NAME = "Groupe_2"
DATASET_NAME = "⭐️ cnam_product_2024"
DATASET_VERSION = "initial"
EXPERIMENT_NAME = "yolo_training_experiment"

# Initialisation du client Picsellia
client = Client(api_token=API_TOKEN, organization_name=ORGANIZATION_NAME)

# Télécharger le dataset depuis Picsellia si le dossier datasets n'existe pas
dataset = client.get_dataset(DATASET_NAME).get_version(DATASET_VERSION)
if not os.path.exists("./datasets"):
    dataset.list_assets().download("./datasets")

# Initialisation du projet et création de l'expérience
project = client.get_project(PROJECT_NAME)
experiment = project.get_experiment(EXPERIMENT_NAME)

# Chemin pour les annotations exportées
annotation_output_path = "./datasets/annotations"

# Exporter les annotations au format YOLO
dataset.export_annotation_file(AnnotationFileType.YOLO, annotation_output_path)

# Recherche du fichier ZIP dans l'arborescence des annotations
def find_zip_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                return os.path.join(root, file)
    return None

# Trouver le fichier ZIP
zip_file_path = find_zip_file(annotation_output_path)

if zip_file_path:
    # Décompression du fichier ZIP
    base_dir = "./datasets"
    annotations_dir = os.path.join(base_dir, "annotations")

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(annotations_dir)

    # Supprimer le fichier ZIP après extraction
    os.remove(zip_file_path)
else:
    print("Aucun fichier ZIP trouvé dans l'arborescence des annotations.")

# Trouver le fichier yaml dans l'arborescence des annotations
def find_yaml_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                return os.path.join(root, file)
    return None

# Trouver le fichier data.yml
data_yaml_path = find_yaml_file(annotations_dir)

# Charger les classes depuis data.yml
class_names = []
if data_yaml_path:
    with open(data_yaml_path, "r") as yaml_file:
        data_content = yaml.safe_load(yaml_file)
        class_names = data_content.get("names", [])
else:
    print("Fichier data.yml introuvable. Veuillez vérifier les annotations.")
    exit(1)



# Structuration des données pour YOLO
output_dir = "./datasets/structured"
images_dir = f"{output_dir}/images"
labels_dir = f"{output_dir}/labels"
train_dir = "train"
val_dir = "val"
test_dir = "test"
split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

# Créer les répertoires de sortie
os.makedirs(f"{images_dir}/{train_dir}", exist_ok=True)
os.makedirs(f"{images_dir}/{val_dir}", exist_ok=True)
os.makedirs(f"{images_dir}/{test_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{train_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{val_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{test_dir}", exist_ok=True)

# Liste des fichiers d'images et d'annotations
image_files = [f for f in os.listdir(base_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
label_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

image_to_label = {
    img: img.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
    for img in image_files
}

paired_files = [(img, lbl) for img, lbl in image_to_label.items() if lbl in label_files]

# Répartition des données en train/val/test
random.shuffle(paired_files)
n_total = len(paired_files)
n_train = int(n_total * split_ratios["train"])
n_val = int(n_total * split_ratios["val"])

splits = {
    "train": paired_files[:n_train],
    "val": paired_files[n_train : n_train + n_val],
    "test": paired_files[n_train + n_val :],
}

# Déplacer les fichiers vers les répertoires correspondants
for split in splits:
    for img, lbl in splits[split]:
        img_path = os.path.join(base_dir, img)
        lbl_path = os.path.join(annotations_dir, lbl)
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            shutil.move(img_path, f"{images_dir}/{split}/{img}")
            shutil.move(lbl_path, f"{labels_dir}/{split}/{lbl}")
        else:
            print(f"Fichier manquant pour l'image {img} ou l'annotation {lbl}.")

# Génération du fichier config.yaml pour YOLO
config = {
    "path" : "./datasets/structured",
    "train": "/Users/isamet/git/ia/ia-cours-4/datasets/structured/images/train",
    "val": "/Users/isamet/git/ia/ia-cours-4/datasets/structured/images/val",
    "test": "/Users/isamet/git/ia/ia-cours-4/datasets/structured/images/test",
    "nc": len(class_names),
    "names": class_names,
}

config_path = "./config.yaml"
with open(config_path, "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

# Initialisation du modèle YOLO
model = YOLO("yolo11n.yaml")  # Utiliser la configuration YOLOv8n par défaut

# Entraînement du modèle
model.train(
    data=config_path,
    epochs=1,
    imgsz=640,
    project="./train_results",  # Gérer les résultats dans ce dossier
    name="yolo_model",  # Nom du modèle pour les résultats
    device='mps',  # Utilisation du GPU Metal sur Mac
)

# Envoi des artefacts et métriques à Picsellia
trained_model_path = "./train_results/yolo_model/weights/best.pt"

print("Entraînement terminé et artefacts envoyés à Picsellia.")
