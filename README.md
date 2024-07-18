
<h1 align="center">
<img src="https://github.com/user-attachments/assets/ceffbe1c-a0b2-4426-a145-ca3b00256676" alt="handSign">
</h1>

# Hand Sign Language Recognition with Pytorch and MediaPipe

## Introduction 

This project presents an innovative hand sign recognition system using deep learning and computer vision. By leveraging a ResNet34 model - Pytorch combined with MediaPipe and OpenCV, the system accurately detects and interprets hand signs for the alphabet and numbers in real-time. This cutting-edge technology promises to enhance accessibility and revolutionize human-computer interaction, offering new possibilities for communication and control.

## Demo
**Hand gesture**
<p align="center">
<img src="https://github.com/user-attachments/assets/5ae4c991-08b1-40c1-8ae3-2f1bd7075008"  height="350" alt="handSign">
</p>

**Working**
Press "w" on the keyboard to write result
<p align="center">
<img src="https://github.com/user-attachments/assets/6eb1b2ef-d8e7-45c4-be79-25a701b35a4e" height= "400" alt="handSign">
</p>

**Collect data**
1. Select a hand sign to collect.
2. Press "s" on the keyboard to take screenshots.
( Images will be saved in a folder of the same hand sign name )
<p align="center">
<img src="https://github.com/user-attachments/assets/e6c702f7-2f50-466e-9620-fd3019ac3ed4"   height= "400"  alt="handSign">
</p>

## Requirements
**Run with IDE**
cuda 12.2 is used in this project
```
* opencv-python==4.5.5.62
* matplotlib==3.8.3
* numpy==1.26.4
* mediapipe==0.10.14
* scikit-learn==1.4.1.post1
* tensorboard==2.16.2
```
```
pip install -r requirements.txt
```
**Run with Docker**
```
docker build -t <imageName> .
docker run -it --gpus all <imageName>
```
## Issues 
This is a problem you may encounter when running the program.
-   Before running `handSignDetection.py`, you must run `train.py` to train the model and save the `best.pt` checkpoints.
-   `train.py` should be run for about 15 epochs to improve the model.
-   You can collect more data to continue training the model.
