# IA COURS 4 - Projet

## Technologies utilisées
* Version Python : 3.11 🐍
* Formatter : Black ⚫
* Linter : Pylint 🧹
* Static type checker : Pyright 🔍
* Outils de gestion de dépendances : Poetry 📦
* Outils de gestion des pre-commit hooks : Pre-commit 🎣

## Description
Ce projet a pour objectif d'entrainer un modèle d'IA en Python capable de détecter des produits comme des mikado, des kinder pinguin, des bouteilles en plastique, etc..., sur des images, des vidéos ou avec une caméra.

Le projet comprend deux pipelines :
* **Pipeline de training** : utilisation d’un dataset annoté sur [Picsellia](https://www.picsellia.com/) pour entraîner un modèle de détection d'objets avec les modèles YOLO d'[Ultralytics](https://www.ultralytics.com/fr).
* **Pipeline d'inférence** : charge le modèle entraîné pour effectuer des détections sur des images, vidéos ou via une webcam.

Picsellia est utilisé pour l'annotation des datasets et le suivi des expérimentationss.
Ultralytics et ses modèles YOLO sont choisis pour la détection d'objets.

## Installation
```shell
git clone https://github.com/joe67100/ia-cours-4.git
cd ia-cours-4
poetry install

# Si poetry n'est pas installé :
# pip install poetry
```

## Utilisation

Pour lancer une session d'entraînement, utilisez la commande suivante:
```shell
python main.py train [-h] --dataset_version DATASET_VERSION --project_name PROJECT_NAME
```
Example:
```shell
python main.py train --dataset_version initial --project_name Groupe_2
```
---

Pour lancer une session d'inférence, utilisez la commande suivante:
```shell
python main.py infer [-h] --model MODEL --model_version MODEL_VERSION {video,image,camera}
```
Examples:
```shell
# For video
python main.py infer --model Groupe_2 --model_version Groupe_2-55 video "C:\videoplayback.mp4"
```
```shell
# For image
python main.py infer --model Groupe_2 --model_version Groupe_2-55 image "C:\image.jpg"
```
```shell
# For camera
python main.py infer --model Groupe_2 --model_version Groupe_2-55 camera
```

## Liens utiles
- [Picsellia : Projects / Groupe_2](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936420-552b-796d-a41c-3b3bf1f7348f)
- [**Meilleur model_version**](https://)


## Auteurs

<table style="width:100%; text-align:center;">
  <tr>
    <td><a href="https://github.com/joe67100"><img src="https://avatars.githubusercontent.com/u/71235356?v=4?s=100" alt="Joe67100's profile picture" /></a></td>
    <td><a href="https://github.com/Bricklou"><img src="https://avatars.githubusercontent.com/u/15181236?v=4?s=100" alt="Bilou's profile picture" /></a></td>
    <td><a href="https://github.com/AIsamet"><img src="https://avatars.githubusercontent.com/u/94604758?v=4?s=100" alt="Isamet's profile picture" /></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/joe67100">Joé</a>
    <td><a href="https://github.com/Bricklou">Bilou</a>
    <td><a href="https://github.com/AIsamet">Isamet</a>
  </tr>
</table>
