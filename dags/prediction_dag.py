from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from src.preprocessing.data_preprocessing import data_preprocessing
from datetime import datetime, timedelta


def load_data():
    with open('params.yaml') as file:
        try:
            params = yaml.safe_load(file)['preprocessing']
        except yaml.YAMLError as exception:
            print(exception)

    data_preprocessing(
        dataset_csv_path='data/test.csv',
        save_path='data/preprocessed/train_dataset.csv',
        train=False,
        dataset_for_lag_features_csv_path='data/sales_train.csv',
        date_block_num=34,
        description_csv_path='data/'
    )

default_args = {
    'owner': 'coder2j',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    #default_args=default_args,
    dag_id='our_dag_with_python_operator_v07',
    description='Dag for test dataset prediction.',
    start_date=datetime(2021, 10, 6),
    schedule_interval='@daily'
) as dag:
    task1 = PythonOperator(
        task_id='get_name',
        python_callable=load_data
    )
