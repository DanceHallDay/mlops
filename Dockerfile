FROM python:3.8-slim-buster

WORKDIR /app


COPY . .
RUN pip install -r requirements.txt


RUN dvc remote modify myremote gdrive_use_service_account true
RUN dvc remote modify myremote --local gdrive_service_account_json_file_path service_account_json/service_account_key.json
RUN dvc pull
