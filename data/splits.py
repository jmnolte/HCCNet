import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class GroupStratifiedSplit(GroupShuffleSplit):

    def __init__(
            self,
            split_ratio: float = 0.8,
            n_splits: int = 100
        ) -> None:
        super().__init__(
            train_size=split_ratio, 
            n_splits=n_splits
        )
        
        '''
        Args:
            split_ratio (float): Ratio of the data to be used for training.
            n_splits (int): Number of splits to perform.
        '''
        self.test_ratio = 1 - split_ratio

    def select_best_split(
            self,
            dataframe: pd.DataFrame,
            splits: list
            ) -> pd.DataFrame:

        '''
        Args:
            dataframe (pd.DataFrame): Dataframe to split.
            splits (list): List of splits.
        '''
        label_ratio = 1
        for split in splits:
            set1, set2 = dataframe.iloc[split[0]], dataframe.iloc[split[1]]
            if abs(set1['label'].mean() - set2['label'].mean()) < label_ratio:
                if len(set1) / (1 - self.test_ratio) > len(set2) / self.test_ratio * 0.95 and len(set1) / (1 - self.test_ratio) < len(set1) / self.test_ratio * 1.05:
                    label_ratio = abs(set1['label'].mean() - set2['label'].mean())
                    best_split = split
        return best_split
    
    def split_dataset(
            self,
            dataframe: pd.DataFrame,
            ) -> pd.DataFrame:
        
        '''
        Args:
            dataframe (pd.DataFrame): Dataframe to split.
        '''
        splits = super().split(dataframe, groups=dataframe['patient_id'])
        split1, split2 = self.select_best_split(dataframe, splits)
        return dataframe.iloc[split1], dataframe.iloc[split2]