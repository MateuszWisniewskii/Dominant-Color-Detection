import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from colorConversion import RGBtoLAB
import neuralNetwork
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from skimage.color import rgb2lab


#class for creating dataset for pyTorch
class NNDataSet(Dataset):
    def __init__(self, csv_file, images_folder, label_type="ground_truth", transform=None):
        #load csv file
        self.df = pd.read_csv(csv_file)

        self.images_folder = images_folder

        self.transform = transform

        if label_type == "clusters":
            self.labels = self.df.iloc[:, 16:].values.astype(np.float32)
        elif label_type == "ground_truth":
            self.labels = self.df.iloc[:, 1:16].values.astype(np.float32)
        else:
            print("Error selecting dataset type type")

        self.target_height = 480
        self.target_width = 640
            

    #function that returns number of samples in dataset
    def __len__(self):
        return len(self.df)
    
    #function that returns one sample from dataset
    def __getitem__(self, id):
        image_name = self.df.iloc[id, 0]
        image_path = f"{self.images_folder}/{image_name}"

        image_lab = RGBtoLAB(image_path)

        if self.transform:
            image_lab = self.transform(image_lab)
        
        c, h, w = image_lab.shape
        pad_h = self.target_height - h
        pad_w = self.target_width - w

        # Padujemy tylko jeśli trzeba (padding: (left, right, top, bottom))
        padding = (0, pad_w, 0, pad_h)
        image_lab_padded = F.pad(torch.tensor(image_lab), padding, mode='constant', value=0)


        labels = self.labels[id]
        print(f"getitem {id}: image_lab shape {image_lab.shape}, label shape {self.labels[id].shape}")
        return image_lab_padded.float(), torch.tensor(labels)


#TODO: change this to match data loader but first change NN
if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser(description="Script for training neural network. Can only be used after generating csv file.")
    parser.add_argument("--dataset", type=str, required=True, help="Earlier generated csv file.")
    parser.add_argument("--dataset_type", type=str, required=True, help="Select between clusters and ground_truth")
    parser.add_argument("--images_folder", type=str, required=True, help="Path to directory with images")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of data batch per epoch.")


    args = parser.parse_args()

    #creation of dataset
    data_set = NNDataSet(args.dataset, args.images_folder, args.dataset_type)
    #loader creation
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

    #init neural network
    model = neuralNetwork.ConvolutionalNeuralNetwork()

    #define the loss function
    criterion = nn.MSELoss()

    #Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    #list for loss per epoch
    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            #forward propagation
            outputs = model(images)
            #calculate loss 
            loss = criterion(outputs, labels)
            #reset gradients
            optimizer.zero_grad()
            #backward propagation
            loss.backward()
            #update weights
            optimizer.step()
            #accumulate loss
            epoch_loss += loss.item()
        #calucated average loss
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    #save trained model in file
    torch.save(model.state_dict(), "model.pt")

    #print tarining graph
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss over Epochs")
    plt.show()





    
