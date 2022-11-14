from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from collections import defaultdict
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
#import neptune.new as neptune
import gc
import yaml
import json

from src.preprocessing.Dataset import *
from src.preprocessing.CrossValidation import *
import matplotlib.pyplot as plt

def cross_validation(X: pd.DataFrame, cv: ICrossValidation, cat_features_list : List[str], *args, **kwargs) -> None:
    rmse = defaultdict(list)
    feature_importences = []
    
    for train_idx, valid_idx in cv.split(X, 'date_block_num'):

        curr_X_train = X.loc[train_idx].drop(['item_cnt_day'], axis=1)
        curr_y_train = X.loc[train_idx, 'item_cnt_day']

        curr_X_valid = X.loc[valid_idx].drop(['item_cnt_day'], axis=1)
        curr_y_valid = X.loc[valid_idx, 'item_cnt_day']

        train_dataset = Pool(
            curr_X_train,
            curr_y_train,
            cat_features=cat_features_list,
        )

        valid_dataset = Pool(
            curr_X_valid,
            curr_y_valid,
            cat_features=cat_features_list,
        )
        
        curr_model = CatBoostRegressor(
            *args,
            **kwargs
        )
        curr_model.fit(train_dataset, eval_set=valid_dataset)

        rmse['train'].append(
            curr_model.get_best_score()['learn']['RMSE']
        )
        rmse['test'].append(
            curr_model.get_best_score()['validation']['RMSE']
        )

        feature_importences.append(
            curr_model.get_feature_importance()
        )

        del curr_X_train, curr_y_train, curr_X_valid, curr_y_valid
        gc.collect()

    return rmse, np.mean(feature_importences, axis=0)


def save_metrics(X ,rmse, feature_importances):
    rmse_mean = {
        'train' : np.mean(rmse['train']),
        'test' : np.mean(rmse['test'])
    }
    with open('evaluations/metrics/metrics.json', 'w') as f:
        json.dump(rmse_mean, f)

    pd.DataFrame(
        rmse['train'],
        columns=['train']
    ).to_csv('evaluations/plots/plots_train.csv')

    pd.DataFrame(
        rmse['test'],
        columns=['test']
    ).to_csv('evaluations/plots/plots_test.csv')

    sorted_idx = np.argsort(feature_importances)
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.savefig("evaluations/images/fi.png",dpi=80)

        


if __name__ == '__main__':
    with open('params.yaml') as file:
        try:
            params = yaml.safe_load(file)
            model_params = params['model_params']
            categorical_features = params['categorical_features']
            cross_validation_params = params['cross_validation']
        except yaml.YAMLError as exception:
            print(exception)
    X = pd.read_csv('data/preprocessed_data/train_dataset.csv')

    rmse, feature_importances = cross_validation(
        X, 
        SlidingWindowCV(**cross_validation_params), 
        cat_features_list=categorical_features, 
        **model_params
    )

    save_metrics(rmse, feature_importances)
