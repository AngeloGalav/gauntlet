FROM python:3.10-slim

WORKDIR /app

COPY . /app
COPY ./container_requirements.txt /app

RUN pip install flask flask-cors
RUN pip install -r container_requirements.txt

CMD ["python", "server.py"]