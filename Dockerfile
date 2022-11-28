FROM apache/airflow:latest-python3.9
#FROM python:3.8-slim-buster


#WORKDIR /app

#RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
#RUN apt-get update && apt-get install -y dvc

COPY . .

RUN pip install -r requirements.txt
RUN pip install dvc
RUN pip install 'dvc[gdrive]'

RUN dvc remote modify myremote gdrive_use_service_account true
RUN dvc remote modify myremote --local gdrive_service_account_json_file_path service_account_json/service_account_key.json

RUN dvc pull