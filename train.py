#!/usr/bin/python3
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from tqdm import tqdm
from data import DataSet
from os.path import exists


DEVICE="cuda:0"
DATA="new"

class TrainModel:
    def __init__(
        self,
        net,
        EPOCHS=10,
        BATCH_SIZE=256,
        optimizer_state=None,
        loss=None,
        save=None,
    ):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.net = net
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
        data_idx = 0
        while exists(f"{DATA}.{data_idx}.npz"):
            data = np.load(f"{DATA}.{data_idx}.npz", allow_pickle=True)
            X, Y = data["arr_0"], data["arr_1"]
            dataloader = torch.utils.data.DataLoader(DataSet(X, Y), shuffle=True, batch_size=self.BATCH_SIZE)
            losses = []
            idx = 0
            loss = 0
            for epoch in tqdm(range(self.EPOCHS)):
                for batch_X, batch_Y in tqdm(dataloader):
                    self.net.zero_grad()
                    self.optimizer.zero_grad()
                    outputs = self.net(batch_X)

                    loss = self.loss_function(outputs.reshape([self.BATCH_SIZE]), batch_Y)
                    losses.append(float(loss))

                    loss.backward()
                    self.optimizer.step()
                    idx += 1
                print(loss)
                data_idx += 1
            print(f"Data idx: {data_idx}")
        return loss
    
if __name__ == "__main__":
    net = Net().to(DEVICE)
    TrainModel(net)
