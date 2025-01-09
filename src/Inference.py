class Inference:
    def __init__(self, mode: str, model: str, model_version: str):
        self.mode = mode
        self.model = model
        self.model_version = model_version
        pass

    def infer(self) -> None:
        print(self.mode)
        print(self.model)
        print(self.model_version)
