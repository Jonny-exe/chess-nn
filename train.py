#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from tqdm import tqdm

DEVICE="cuda:0"

class TrainModel:
    def __init__(
        self,
        net,
        data,
        STARTING_EPOCHS=0,
        EPOCHS=2,
        # BATCH_SIZE=300,
        BATCH_SIZE=64,
        optimizer_state=None,
        loss=None,
        save=None,
        GRAPH=False,
    ):
        self.EPOCHS = EPOCHS
        self.STARTING_EPOCHS = STARTING_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.GRAPH = GRAPH

        self.net = net
        self.data = data
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

        self.loss_functions = nn.MSELoss().to(DEVICE)

        loss = self.train()
        data_to_save = {
            "net": self.net.state_dict(),
            "epochs": self.EPOCHS,
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
        }
        try:
            torch.save(
                data_to_save, f"models/model{EPOCHS}.pth" if save is None else save
            )
        except FileNotFoundError:
            os.mkdir("models", mode=0o666)
            print("Created folder models")
            torch.save(
                data_to_save, f"models/model{EPOCHS}.pth" if save is None else save
            )

    def train(self):
        losses = []
        idx = 0
        loss = 0
        for epoch in tqdm(range(self.STARTING_EPOCHS, self.EPOCHS, 1)):
            for i in tqdm(range(0, len(self.data), self.BATCH_SIZE)):

                batch_Y = torch.Tensor([y[1] for y in self.data[i:i+self.BATCH_SIZE]]).to(DEVICE)
                batch_X = torch.Tensor([x[0] for x in self.data[i:i+self.BATCH_SIZE]]).to(DEVICE).reshape([-1, 1, 8, 8])
                self.net.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.net(batch_X)  # Shape: (4, -1, 100)
                assert outputs[0] <= 1 and outputs[0] >= -1

                loss = self.loss_functions(outputs[0], batch_Y)

                losses.append(float(loss))

                loss.backward()
                self.optimizer.step()
                idx += 1
            print(loss)

        plt.plot(losses)
        plt.show()
        return loss

    
if __name__ == "__main__":
    data = np.load("data.npy", allow_pickle=True)
    net = Net().to(DEVICE)
    TrainModel(net, data)
