import os
from picsellia import Client
from ultralytics import YOLO
import cv2


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

        yolo_model = YOLO(self.model_file_path)

        if self.mode == "image":
            if not self.file_path:
                raise ValueError("Missing file path")
            if not os.path.exists(self.file_path):
                raise FileExistsError("Invalid file path")

            self._infer_image(yolo_model, self.file_path)
        elif self.mode == "video":
            if not self.file_path:
                raise ValueError("Missing file path")
            if not os.path.exists(self.file_path):
                raise FileExistsError("Invalid file path")
            
            self._infer_video(yolo_model, self.file_path)
        elif self.mode == "camera":
            self._infer_webcam(yolo_model)
        else:
            raise Exception("Unknown source mode")

    def _infer_image(self, model: YOLO, file_path: str) -> None:
        [result] = model(file_path)

        result.show()

    def _infer_video(self, model: YOLO, file_path: str) -> None:
        [result] = model(file_path)

        result.show()

    def _infer_webcam(self, model: YOLO) -> None:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run yolo inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO inference (q to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
