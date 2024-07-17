import torch
import torch.nn as nn
import argparse
from torchvision.models import resnet34
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import cv2

def get_args():
    #built arguments
    parser = argparse.ArgumentParser(description="inference NN model")
    #add argument
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="train/checkpoints")
    args = parser.parse_args()
    return args

class PredictHandSign():
    def __init__(self):
        # create device to access GPU and model, epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # classes
        self.classes = ["A", "B", "C", "I", "<3", "U"]
        # softmax
        self.softmax = nn.Softmax()
        self.args = get_args()
        self.model = self.loadModel()
    def loadModel(self): #Load model
        model = resnet34()
        model.fc = nn.Linear(in_features=512, out_features=6)
        # take the best model - load_state_dict
        checkpoint = torch.load(os.path.join(self.args.checkpoint_path, "best.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model
    def normalizeImage(self, img): #Normailize Image
        image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        image = cv2.resize(image, (self.args.image_size, self.args.image_size))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image / 255.
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))[None, :, :, :]  # add batch size
        image = torch.from_numpy(image).float()
        return image
    def predict(self, frame):
        #Normalize image
        image = self.normalizeImage(frame).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            prob = self.softmax(output)
            # tacke class prediction
            predicted_prob, predicted_class = torch.max(prob, dim=1)
            # check level believe of model with class
            score = predicted_prob[0].item() * 100

        return self.classes[predicted_class[0].item()]


if __name__ == '__main__':
    # create device to access GPU and model, epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classes
    classes = ["A", "B", "C", "I", "<3", "U"]
    softmax = nn.Softmax()
    args = get_args()
