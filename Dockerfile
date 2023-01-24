FROM python:3.7-slim-buster AS final

COPY predictors requirements.txt /mnt/spotad/inprocess/
COPY parameters.conf /mnt/spotad/inprocess/files/
WORKDIR /mnt/spotad/inprocess

RUN pip install -r requirements.txt

CMD ["python", "./train-model.py"]
