#!/usr/bin/python3
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from tqdm import tqdm
from data import DataSet

DEVICE="cuda:0"

class TrainModel:
    def __init__(
        self,
        net,
        data,
        EPOCHS=10,
        BATCH_SIZE=256,
        optimizer_state=None,
        loss=None,
        save=None,
    ):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.net = net
        self.data = data
        self.dataloader = torch.utils.data.DataLoader(DataSet(data), shuffle=True, batch_size=self.BATCH_SIZE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.1)

        self.loss_function = nn.MSELoss().to(DEVICE)

        loss = self.train()
        data_to_save = {
            "net": self.net.state_dict(),
            "epochs": self.EPOCHS,
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(
            data_to_save, f"model.pth" if save is None else save
        )

    def train(self):
        losses = []
        idx = 0
        loss = 0
        for epoch in tqdm(range(self.EPOCHS)):
            for batch_X, batch_Y in tqdm(self.dataloader):
                self.net.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.net(batch_X)

                loss = self.loss_function(outputs.reshape([-1]), batch_Y)
                losses.append(float(loss))

                loss.backward()
                self.optimizer.step()
                idx += 1
            print(loss)
        return loss

    
if __name__ == "__main__":
    data = np.load("data.npy", allow_pickle=True)
    net = Net().to(DEVICE)
    TrainModel(net, data)
