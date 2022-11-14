from abc import ABC

class IDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    def get_data(self):
        pass

    def get_labels(self):
        pass 