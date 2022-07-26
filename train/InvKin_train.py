import pickle
import torch
from Model.model import InvKin
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class CustomDataSet(Dataset):
    def __init__(self,numpy_x_name,numpy_y_name,dir):
        self.x = np.load(dir+numpy_x_name)
        self.y = np.load(dir+numpy_y_name)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
dataset1 = CustomDataSet('InvKin_x.npy','InvKin_y.npy', '../IGL_data/')

grid_lr    = [0.0001, 0.00005, 0.00001]
grid_wd    = [1e-4,1e-5,1e-6]
grid_batch = [20000,40000,60000]

for x,batch in enumerate(grid_batch):
    train_loader1 = DataLoader(dataset1, shuffle = True,batch_size = batch)

    epochs = 100
    state_dim = 4
    device = "cuda"
    agent=InvKin(state_dim,device)
    agent.to(device)
    for y,lr in enumerate(grid_lr):
        for z, wd in enumerate(grid_wd):
            optimizer = torch.optim.Adam(agent.parameters(), lr=lr,weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.9)

            agent.train()
            loss = nn.MSELoss()
            for i in range(epochs):
                temp_loss1 = 0
                temp_loss2 = 0
                for k,(state,label) in enumerate(train_loader1):
                    optimizer.zero_grad()
                    output=agent(state.type(torch.FloatTensor).to(device))
                    loss_ = loss(label.type(torch.FloatTensor).to(device),output)
                    loss_.backward()
                    optimizer.step()
                    temp_loss1 += loss_.item()
                print("========="+str(i)+"=========")
                print(temp_loss1)
            torch.save(agent.state_dict(), '../model_save/InvKin'+str(x)+str(y)+str(z))