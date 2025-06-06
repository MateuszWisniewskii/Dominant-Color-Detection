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


class NNDataSet(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.real_values = df[["L", "a", "b"]].values.astype(np.float32)
        self.images = df.iloc[:, 4:].values.astype(np.float32)
        self.images = self.images.reshape((-1, 3, 64, 64))

    def __len__(self):
        return len(self.real_values)
    
    def __getitem__(self, id):
        x = torch.tensor(self.images[id])
        y = torch.tensor(self.real_values[id])
        return x, y




if __name__ == "__main__":
    createDataCSV("data.csv")

    data_set = NNDataSet("data.csv")
    loader = DataLoader(data_set, batch_size=16, shuffle=True)

    model = neuralNetwork.ConvolutionalNeuralNetwork()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 1000

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss over Epochs")
    plt.show()





    
