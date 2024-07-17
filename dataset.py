from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os
import cv2
import matplotlib.pyplot as plt
import shutil
from torchvision.transforms import Compose, ToTensor, ColorJitter, Resize, RandomAffine, Normalize

# Define a custom dataset class for animal images
class HandSignDataset(Dataset):
    def __init__(self, root="data", is_train=True, transform=None):
        # List of categories (HandSign classes) in the dataset
        self.categories = ["A", "B","C", "I", "<3", "U"]
        if is_train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "test")
        # Initialize lists to store image paths and corresponding labels
        self.images = []
        self.labels = []
        # Loop over each category to collect image paths and their labels
        for idx, cat in enumerate(self.categories):
            catPath = os.path.join(root, cat)
            imgPaths = os.listdir(catPath)
            for img in imgPaths:
                self.images.append(os.path.join(catPath, img))
                self.labels.append(idx)
        # Store the transformation function (if any)
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        Retrieve a sample from the dataset.
        Parameters:
        - item: int, the index of the sample to retrieve.
        Returns:
        - A tuple (image, label), where image is the transformed image and label is the category index.
        """
        # Open the image file and convert it to RGB mode
        img = cv2.imread(self.images[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get the label for the image
        label = self.labels[item]
        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)
        # Return the image and its label
        return img, label

    def checkDir(self, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


if __name__ == '__main__':

    train_transform = Compose([
        ToTensor(),

    ])  # normalize images
    train_dataset = HandSignDataset(is_train=True, transform=train_transform)
    a,b = train_dataset[10]
    print(train_dataset.__len__())
    print(b)
    plt.imshow( a.permute(1, 2, 0)  )
    plt.waitforbuttonpress()
    train_dataloader = DataLoader(
            dataset= train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
