from src.preprocessing.ICrossValidation import *


class SlidingWindowCV(ICrossValidation):
    def __init__(self, train_size, test_size):
        self.train_size = train_size
        self.test_size = test_size

    def split(self, df : pd.DataFrame, split_column : str):
        #df = df.sort_values(by=split_column).reset_index()
        border_indexes = df.loc[df[split_column]!= df[split_column].shift()].index.tolist()
        
        for i in range(34 - (self.train_size + self.test_size)):
            train_indexes = np.array(
                [
                    *range(
                        border_indexes[i],
                        border_indexes[i + self.train_size]
                    )
                ]
            )
            test_indexes = np.array(
                [
                    *range(
                        border_indexes[i + self.train_size], 
                        border_indexes[i + self.train_size + self.test_size]
                    )
                ]
            )
            yield train_indexes, test_indexes        
            
            
class TrainTestCV(ICrossValidation):
    def __init__(self, border_value_train : int, border_value_test : int):
        self.border_value_train = border_value_train
        self.border_value_test = border_value_test
    
    def split(self, df : pd.DataFrame, split_column : str):
        if (self.border_value_train>= df[split_column].unique().shape[0] and self.border_value_train>= df[split_column].unique().shape[0]):
            raise IndexError("Border month can't be bigger that number of monthes")
            
        df = df.sort_values(by=split_column).reset_index()
        train_indexes = df.loc[
            df[split_column] < self.border_value_train
        ].index.tolist()
        test_indexes = df.loc[
            (df[split_column] >= self.border_value_train) &
            (df[split_column] < self.border_value_test)
        ].index.tolist()
        
        yield train_indexes, test_indexes