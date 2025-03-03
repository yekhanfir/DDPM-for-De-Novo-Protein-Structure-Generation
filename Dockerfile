FROM pytorch/pytorch

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && pip install -r requirements.txt
