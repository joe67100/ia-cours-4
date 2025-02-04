import os
from picsellia import Client
from ultralytics import YOLO
import cv2
import time


class Inference:
    def __init__(
        self,
        client: Client,
        mode: str,
        model: str,
        model_version: str,
        file_path: str | None = None,
        confidence_threshold: float = 0.7,
        frame_delay: float = 0.1,
    ):
        self.client = client
        self.mode = mode
        self.model = model
        self.model_version = model_version
        self.file_path = file_path
        self.confidence_threshold = confidence_threshold
        self.frame_delay = frame_delay

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
        results = model(file_path)
        self._filter_and_display(results)

    def _infer_video(self, model: YOLO, file_path: str) -> None:
        results = model(file_path)
        self._filter_and_display(results)

    def _infer_webcam(self, model: YOLO) -> None:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            filtered_frame = self._filter_results(results, frame)

            cv2.imshow("YOLO inference (q to quit)", filtered_frame)
            time.sleep(self.frame_delay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _filter_results(self, results, frame):
        annotated_frame = frame.copy()
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()
                if confidence >= self.confidence_threshold:
                    annotated_frame = result.plot()
        return annotated_frame

    def _filter_and_display(self, results):
        for result in results:
            result.boxes = [
                box
                for box in result.boxes
                if box.conf[0].item() >= self.confidence_threshold
            ]
            result.show()
