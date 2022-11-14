import numpy as np
import pandas as pd
import re
from itertools import product
from typing import List, Tuple
from src.preprocessing.IDataset import *
from src.preprocessing.memory_reduce import reduce_memory_usage

def setdiff2d_set(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2))

class Dataset(IDataset):
    def __init__(self, csv_path:str, description_files_path:str, train:bool=False, *args, **kwargs) -> None:
        self.train = train
        self.description_files_path = description_files_path
        self.df = self.__etl__(csv_path, *args, **kwargs)
        super().__init__()

    def get_data(self, add_lag_features:bool=False, *args, **kwargs) -> pd.DataFrame:
        result_df = self.df.copy()
        
        if add_lag_features:
            result_df = self.__add_lag_features__(result_df, *args, **kwargs)
            
        if self.train:
            result_df =  result_df.drop(['item_cnt_day', 'item_price', 'item_revenue'], axis=1)
        result_df['month'] = result_df['date_block_num'].apply(lambda x : x % 12)
        #result_df['year'] = result_df['date_block_num'].apply(lambda x : x // 12)
        #result_df.drop(['date_block_num'], axis=1, inplace=True)
        result_df['date_block_num'] = result_df['date_block_num'].astype(np.int16)
        result_df['shop_id'] = result_df['shop_id'].astype(np.int16)
        result_df['item_id'] = result_df['item_id'].astype(np.int16)
        result_df['month'] = result_df['month'].astype(np.int16)
        #result_df['year'] = result_df['year'].astype(np.int16)
        
        return result_df.fillna(0)

    def get_labels(self) -> pd.Series:
            return self.df.item_cnt_day if self.train else None

    def __fix_category__(self, category_id:int) -> str:
        if category_id == 0:
            return "Headphones"
        elif category_id in range(1, 8):
            return 'Accessory'
        elif category_id == 8:
            return 'Tickets'
        elif category_id == 9:
            return 'Delivery'
        elif category_id in range(10, 18):
            return 'Consoles'
        elif category_id in range(18, 32):
            return 'Games'
        elif category_id in range(32, 37):
            return 'Pay Card'
        elif category_id in range(37, 42):
            return 'Films'
        elif category_id in range(42, 54):
            return 'Books'
        elif category_id in range(54, 61):
            return 'Music'
        elif category_id in range(61, 73):
            return 'Gifts'
        elif category_id in range(73, 79):
            return 'Soft'
        elif category_id in range(79, 81):
            return 'Music'
        elif category_id in range(81, 83):
            return 'Clean'
        else:
            return 'Charging'
    
    def __name_preprocessing__(self, name : str) -> str:
        name = name.lower()
        #delete addition information(name of street)
        name = name.partition('(')[0]
        name = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', name)
        name = name.replace('  ', ' ')
        name = name.strip()

        return name  

    def __shop_name_preprocessing__(self, df: pd.DataFrame) -> pd.DataFrame: 
        df['shop_name'] = df['shop_name'].apply(self.__name_preprocessing__)
        df['shop_city'] = df['shop_name'].apply(lambda x : x.split(' ')[0])

        df.loc[df['shop_city'] == 'н', 'shop_city'] = 'нижний новгород'
        df.loc[df['shop_name'].str.contains('новгород'), 'shop_name'] = (
            df.loc[df['shop_name'].str.contains('новгород'), 'shop_name'].
            apply(lambda x : x.replace('новгород', ''))
        )

        df['shop_type'] = df['shop_name'].apply(
            lambda x : x.split()[1] if (len(x.split())>1) else 'other'
        )
        df.loc[
            (df['shop_type'] == 'орджоникидзе') |
            (df['shop_type'] == 'ул') |
            (df['shop_type'] == 'распродажа') |
            (df['shop_type'] == 'торговля'),
            'shop_type'
        ] = 'other'

        return df
    
    def __item_category_name_preprocessing__(self, df:pd.DataFrame) -> pd.DataFrame:
        df['item_category'] = (
            df['item_category_name']
            .str.split(' - ').apply(lambda x: x[0])
        )
        df['item_subcategory'] = (
            df['item_category_name']
            .str.split(' - ').apply(lambda x: x[-1])
        )

        df['item_fixed_category'] = df['item_category_id'].apply(self.__fix_category__)
        return df
    

    def __add_lag_features__(self, df:pd.DataFrame, other_df=None, lags:List = [], lag_features:List = [], func_list = ['median', 'sum']) -> pd.DataFrame:
        result_df = df.copy()
        if other_df:
            result_df = pd.concat([result_df, *other_df])
   
        for lag in lags:
                df_lag = (
                    result_df[lag_features + ['date_block_num', 'item_id', 'shop_id']]
                    .copy()
                    .groupby(['date_block_num', 'shop_id', 'item_id'])
                    .agg(
                        {
                            lag_feature : func_list
                            for lag_feature in lag_features}
                    )
                )
                df_lag.columns = pd.Index(
                    [
                        f'{func}_{lag_feature}_lag_{lag}'
                        for lag_feature in lag_features
                        for func in func_list
                    ]
                )
                df_lag.reset_index(inplace=True)

                df_lag['date_block_num'] += lag

                result_df = pd.merge(
                    result_df, 
                    df_lag,
                    on = ['date_block_num', 'item_id', 'shop_id'],
                    how='left'
                )
                
        reduce_memory_usage(result_df)

        return result_df


    def __drop_outliers__(self, df:pd.DataFrame, cat:str) -> pd.DataFrame:
        cat_df = df[df['item_category_id'] == cat]
        cat_df_mean, cat_df_std = cat_df.mean(), cat_df.std()
        cat_df_norm = (cat_df - cat_df_mean) / cat_df_std
        
        cat_df = cat_df[np.abs(cat_df['item_price']) > 3]
        
        return cat_df
    
    def add_column(self, column_name : str, values):
        self.df[column_name] = values
        
    def __add_cartesian_features_product__(self, df, date_block_num : int):
        cartesian_product = np.array(
            list(
                product(
                    df.shop_id.unique(),
                    df.item_id.unique(),
                )
            )
        )
        curr_combinations = df.loc[df.date_block_num == date_block_num ,['shop_id', 'item_id']].to_numpy()
        diff = setdiff2d_set(cartesian_product, curr_combinations)
        
        df = df.append(
            pd.DataFrame(
               np.hstack((diff.tolist(), np.ones((diff.shape[0], 1))*date_block_num)),
                columns=['shop_id', 'item_id', 'date_block_num'],
            )
        )

        return df.fillna(0)
        
        
    def __etl__(self, file_path : str, add_item_cartesian_product : bool = False, date_block_num : int = 34) -> pd.DataFrame:
        dataset = pd.read_csv(file_path)
        
        if not self.train:
            dataset['date_block_num'] = date_block_num
            
        shops = self.__shop_name_preprocessing__(
             pd.read_csv(self.description_files_path + 'shops.csv')
        )
        items = pd.read_csv(self.description_files_path + 'items.csv')
        item_cat = self.__item_category_name_preprocessing__(
            pd.read_csv(self.description_files_path + 'item_categories.csv')
        )
        
        if add_item_cartesian_product:
            dataset = self.__add_cartesian_features_product__(
                dataset,
                30
            )
       
        if self.train:
            dataset['item_revenue'] = dataset['item_price'] * dataset['item_cnt_day']
            dataset.drop(['date'], axis=1, inplace=True)
        
        dataset = pd.merge(dataset, items, on="item_id", how="inner")
        dataset = pd.merge(dataset, shops, on="shop_id", how="inner")
        dataset = pd.merge(dataset, item_cat, on="item_category_id", how="inner")
        
        dataset.drop_duplicates()
        #train_dataset.drop(['date'], axis=1, inplace=True)
            
        #delete neg values in item_price feature
        if self.train:
            dataset = dataset.loc[
                (dataset['item_price'] >= 0) & 
                (dataset['item_cnt_day'] >= 0)
            ]
            #dataset.drop(['date'], axis=1, inplace=True)
        #train_dataset.loc[train_dataset['item_price'] > 0, 'item_price'].apply(np.log)
        
        #drop outlier
        #for cat in tqdm_notebook(train_dataset['item_category_id'].unique()):
            #train_dataset[train_dataset['item_category_id'] == cat] = drop_outliers(train_dataset, cat)
        
        dataset = dataset.sort_values(
            by='date_block_num',
        ).reset_index().drop(['index'], axis=1)
        
        dataset.drop(
            ['shop_name', 'item_name', 'item_category_name'],
            axis=1,
            inplace=True
        )
        
        reduce_memory_usage(dataset)

        return dataset
