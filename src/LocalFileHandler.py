from .Abstract.FileHandler import FileHandler
import os
import zipfile
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class LocalFileHandler(FileHandler):
    def find_file(self, directory: str, extension: str) -> str | None:
        directory_path = Path(directory)
        for file_path in directory_path.rglob(f"*{extension}"):
            return str(file_path)
        return None

    def extract_zip(self, zip_file_path: str, extract_to: str) -> None:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_file_path)
