FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /handstracker
COPY dataset.py ./dataset.py
COPY handSignDetection.py ./handSignDetection.py
COPY HandTrackingModule.py ./HandTrackingModule.py
COPY predict.py ./predict.py
COPY train.py ./train.py
COPY requirements.txt ./requirements.txt
COPY data ./data
COPY train ./train
RUN apt-get update
RUN apt-get install vim ffmpeg libsm6 libxext6  -y
RUN pip install -r ./requirements.txt



