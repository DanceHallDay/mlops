from src.preprocessing.Dataset import *
import yaml
import gc

def data_preprocessing(
    train_dataset_csv_path : str,
    test_dataset_csv_path : str = None,
    description_csv_path : str = None,
    *args,
    **kwargs
) -> None:
    train_dataset = Dataset(
        csv_path = train_dataset_csv_path,
        description_files_path = description_csv_path,
        train = True,
    )

    test_dataset = Dataset(
        csv_path = test_dataset_csv_path,
        description_files_path = description_csv_path,
        train = False,
        date_block_num=34
    )

    ID = test_dataset.get_data()['ID']
    X = train_dataset.get_data(
        other_df = [
            test_dataset.get_data().drop(['ID'], axis=1)
        ],
        *args,
        **kwargs
    )

    y_train = train_dataset.get_labels()

    del train_dataset, test_dataset
    gc.collect()

    X_train = X[:y_train.shape[0]]
    X_test = X[y_train.shape[0]:]

    X_train.loc[:, 'item_cnt_day'] = y_train
    reduce_memory_usage(X_train)
    X_train.to_csv('data/preprocessed/train_dataset.csv')

    reduce_memory_usage(X_test)
    X_test.loc[:, 'ID'] = ID
    X_test.to_csv('data/preprocessed/test_dataset.csv')


if __name__ == '__main__':
    with open('params.yaml') as file:
        try:
            params = yaml.safe_load(file)['preprocessing']
        except yaml.YAMLError as exception:
            print(exception)

    data_preprocessing(
        train_dataset_csv_path='data/sales_train.csv',
        test_dataset_csv_path='data/test.csv',
        description_csv_path='data/',
        **params
    )

    



