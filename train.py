import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  # to ignore warnings
import torch
from torchvision.models.resnet import resnet34, ResNet34_Weights
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, ColorJitter, Resize, RandomAffine, Normalize
from dataset import HandSignDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import shutil
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Pastel1") #change colors here
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
def get_args():
    # Parse command-line arguments
    #Initialize parser
    parser = argparse.ArgumentParser(description="arguments for Hand Sign Training")
    #Adding optional argument
    parser.add_argument("--data_path","-d", type=str, default="data",help="")
    parser.add_argument("--epochs","-e", type=int, default=100,help="")
    parser.add_argument("--batch_size","-b", type=int, default=16,help="")
    parser.add_argument("--num_workers","-w", type=int, default=6,help="")
    parser.add_argument("--num_classes","-n", type=int, default=6,help="")
    parser.add_argument("--image_size","-i", type=int, default=224,help="")
    parser.add_argument("--log_path","-l", type=str, default="train/tensorboard",help="")
    parser.add_argument("--checkpoints_path","-c", type=str, default="train/checkpoints",help="")
    args = parser.parse_args()
    return args

def train(args):
    # Set the device to GPU if available, otherwise use CPU
    devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained ResNet-34 model
    checkpoint = torch.load(os.path.join(args.checkpoints_path,'last.pt'))
    model = resnet34()
    # Replace the final fully connected layer to match our number of classes
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(devide)

    # Define data transformations for training and testing
    train_transform = Compose([
        ToTensor(),
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # normalize images

    test_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])  # normalize images
    # Load the datasets
    train_dataset = HandSignDataset(root= args.data_path,is_train=True, transform=train_transform)
    test_dataset = HandSignDataset(root= args.data_path,is_train=False, transform=test_transform)
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    # Set up TensorBoard writer

    if not os.path.isdir(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    epochs = args.epochs
    writer = SummaryWriter(args.log_path)
    best_acc = -1
    # Training loop
    for epoch in range(current_epoch, epochs):
        # Create progress bar for training
        progress_bar = tqdm(train_dataloader, colour="cyan")
        # Set model to training mode
        model.train()
        # Training phase
        for iter,(images, labels) in enumerate(progress_bar):
            # Move images,labels to device
            images = images.to(devide)
            labels = labels.to(devide)
            output = model(images)
            loss = criterion(output, labels)
            # Update progress bar description
            progress_bar.set_description("Epochs: {}/{} | Iter: {}/{} | Loss: {:0.2f}".format(epoch+1, epochs, iter, len(train_dataloader), loss))
            #writting in dir (Loss/train)
            writer.add_scalar("Loss/train",loss,epoch*len(train_dataloader)+iter)
            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        # Disable gradient computation
        with torch.no_grad():
            for iter, (images, labels) in enumerate(test_dataloader):
                # Move images to device,labels to device
                images = images.to(devide)
                labels = labels.to(devide)
                # Forward pass
                output = model(images)
                # Compute loss
                loss= criterion(output, labels)
                # Get predicted labels
                predictions = torch.argmax(output, dim=1)
                # Collect losses, Collect and predicted labels
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
            # Compute mean loss
            mean_loss = np.mean(all_losses)
            # Compute accuracy
            accuracy= accuracy_score(all_labels, all_predictions)
            # Compute confusion matrix
            conf_matrix= confusion_matrix(all_labels,all_predictions)
            # Print, and write in tensorboard validation results
            print("TEST | epoch:{}/{} | Loss:{:0.4f} | Acc:{:0.4f}".format(epoch + 1, epochs, loss,accuracy))
            # add_scalar test - loss, accuracy
            writer.add_scalar("Test/Loss", loss, epoch)
            writer.add_scalar("Test/Accuracy", accuracy, epoch)
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)

            # Save model checkpoints
            checkpoints ={
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'Acc': accuracy
            }
            # Save last checkpoint
            torch.save(checkpoints, os.path.join(args.checkpoints_path,"last.pt"))
            # Update best accuracy
            if best_acc<accuracy:
                torch.save(checkpoints, os.path.join(args.checkpoints_path, "best.pt"))
                best_acc = accuracy

if __name__ == '__main__':
    args = get_args()
    train(args)
