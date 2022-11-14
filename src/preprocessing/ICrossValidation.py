import numpy as np
import pandas as pd
from abc import ABC

class ICrossValidation(ABC):
    def split(self, df : pd.DataFrame, split_column : str):
        pass