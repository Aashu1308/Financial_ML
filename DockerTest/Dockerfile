FROM docker.io/tensorflow/tensorflow:2.18.0

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app_web.py .

COPY model/ ./model/

EXPOSE 7860

CMD ["python", "app_web.py"]