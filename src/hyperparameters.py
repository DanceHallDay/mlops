import optuna
import catboost as cb
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Dict
import pandas as pd
from src.cross_validation import *
from preprocessing.CrossValidation import *

def objective(trial, X : pd.DataFrame, config : dict, train_w : float = 0.8, test_w : float = 0.2):

    params = {   
        'learning_rate' : trial.suggest_discrete_uniform("learning_rate", 0.001, 0.02, 0.001),
        'depth' : trial.suggest_int('depth', 9, 15),
        'l2_leaf_reg' : trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5),
        'min_child_samples' : trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32]),
        'grow_policy' : 'Depthwise',
        'iterations' : 10,
        'use_best_model' : True,
        'eval_metric' : 'RMSE',
        #'od_type' : 'iter',
        #'od_wait' : 20,
        'used_ram_limit': "3gb",
        'task_type' : 'GPU'
    }

    rmse, _ = cross_validation(
        X, 
        SlidingWindowCV(**config['cross_validation']), 
        cat_features_list=config['categorical_features'], 
        **params#**config['model_params']
    )

    return np.sum(rmse['train']) * train_w + np.sum(rmse['test']) * test_w


def hyperparameters_search(train_dataset_path : str, config : dict, *args, **kwargs):
    X = pd.read_csv(train_dataset_path)
    
    cat_features = config['categorical_features']
    model_params = config['model_params']

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial : objective(trial, X, config, *args, **kwargs), 
        n_trials=1, 
        timeout=600,
        #n_jobs=-1
    )

    config['model_params'].update(study.best_params)

    model = CatBoostRegressor(**config['model_params'])

    train_dataset = Pool(
            X.drop(['item_cnt_day'], axis=1),
            X['item_cnt_day'],
            cat_features=config['categorical_features'],
    )

    model.fit(train_dataset)
    model.save_model('models/model.pickle')

    rmse, feature_importances = cross_validation(
        X, 
        SlidingWindowCV(**config['cross_validation']), 
        cat_features_list=config['categorical_features'], 
        **config['model_params']
    )

    save_metrics(X, rmse, feature_importances)




if __name__ == '__main__':
    with open('params.yaml') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)

    hyperparameters_search('data/preprocessed/train_dataset.csv', config)


