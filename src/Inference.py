import os
from picsellia import Client
from ultralytics import YOLO


class Inference:
    def __init__(
        self,
        client: Client,
        mode: str,
        model: str,
        model_version: str,
        file_path: str | None = None,
    ):
        self.client = client
        self.mode = mode
        self.model = model
        self.model_version = model_version
        self.file_path = file_path

        model = client.get_model(model)
        model_version = model.get_version(model_version)
        self.model_file = model_version.get_file("last_pt")

        self.model_folder_path = f"./models/{model_version.name}"
        self.model_file_path = f"{self.model_folder_path}/{self.model_file.filename}"
        self.model_file.download(self.model_folder_path)

    def infer(self) -> None:
        print(self.mode)
        print(self.model)
        print(self.model_version)

        yolo_model = YOLO(self.model_file_path)

        if self.mode == "image":
            if not self.file_path:
                raise ValueError("Missing file path")
            if not os.path.exists(self.file_path):
                raise FileExistsError("Invalid file path")

            self._infer_image(yolo_model, self.file_path)
        elif self.mode == "video":
            pass
        elif self.mode == "camera":
            pass
        else:
            raise Exception("Unknown source mode")

    def _infer_image(self, model: YOLO, file_path: str):
        [result] = model(file_path)

        result.show()
