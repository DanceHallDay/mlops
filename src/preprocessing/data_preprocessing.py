from src.preprocessing.Dataset import *
import yaml
import gc

def data_preprocessing(
    dataset_csv_path : str,
    save_path : str,
    train : bool,
    date_block_num : int = None,
    dataset_for_lag_features_csv_path : str = None,
    description_csv_path : str = None,
    *args,
    **kwargs
) -> None:
    dataset = Dataset(
        csv_path=dataset_csv_path,
        description_files_path=description_csv_path,
        train=train,
        date_block_num=date_block_num
    )

    if dataset_for_lag_features_csv_path:
        X = dataset.get_data(
            other_df = [
                pd.read_csv(dataset_for_lag_features_csv_path)
            ],
            *args,
            **kwargs
        )

    else:
        X = dataset.get_data(
            *args,
            **kwargs
        )

    if train:
        y = dataset.get_labels()
        X.loc[:, 'item_cnt_day'] = y
    reduce_memory_usage(X)
    X.to_csv(save_path)


if __name__ == '__main__':
    with open('params.yaml') as file:
        try:
            params = yaml.safe_load(file)['preprocessing']
        except yaml.YAMLError as exception:
            print(exception)

    data_preprocessing(
        dataset_csv_path='data/sales_train.csv',
        save_path='data/preprocessed/train_dataset.csv',
        train=True,
        description_csv_path='data/'
    )

    



