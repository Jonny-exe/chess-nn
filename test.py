#!/usr/bin/python3
import torch
import numpy as np
from net import Net
from game import Game
from data import DataSet
if __name__ == "__main__":
    net = Net().to("cuda:0")
    net.load_state_dict(torch.load("model.pth")["net"])
    data = np.load("test.npy", allow_pickle=True)
    dataset = torch.utils.data.DataLoader(DataSet(data), shuffle=True, batch_size=64)
    with torch.no_grad():
        for X, Y in dataset:
            output = net(X)
            print(output)


        

