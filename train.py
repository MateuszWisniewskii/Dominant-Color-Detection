import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from prepareData import loadAndConvertData, createDataCSV
import neuralNetwork
import matplotlib.pyplot as plt
import argparse


#class for creating dataset for pyTorch
class NNDataSet(Dataset):
    def __init__(self, csv_file):
        #load csv file
        df = pd.read_csv(csv_file)
        #extract real values of training images
        self.real_values = df[["L", "a", "b"]].values.astype(np.float32)
        # extract 12288 = 3 * 64 * 64 pixels and reshape to (batch size, channels, size, size)
        self.images = df.iloc[:, 4:].values.astype(np.float32)
        self.images = self.images.reshape((-1, 3, 64, 64))

    #function that returns number of samples in dataset
    def __len__(self):
        return len(self.real_values)
    
    #function that returns one sample from dataset
    def __getitem__(self, id):
        x = torch.tensor(self.images[id])
        y = torch.tensor(self.real_values[id])
        return x, y




if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser(description="Script for training neural network. Can only be used after generating csv file.")
    parser.add_argument("--dataset", type=str, required=True, help="Earlier generated csv file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of data batch per epoch.")

    args = parser.parse_args()

    #creation of dataset
    data_set = NNDataSet(args.dataset)
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





    
