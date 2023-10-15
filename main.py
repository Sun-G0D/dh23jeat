import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os

from os import listdir
from torch.utils.data import TensorDataset, DataLoader

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    AE = AE()
    data = np.load('test1.txt.npy')
    tensor = torch.Tensor(data)
    dataset = TensorDataset(tensor, tensor)
    dataloader = DataLoader(dataset)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(AE.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        for d in dataloader:
            y = AE(d)
            loss = criterion(y, d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 5== 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))