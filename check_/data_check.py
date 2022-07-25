import pickle
import torch
from Model.model import IGL
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
subgoal = '0'
dataset1 = CustomDataSet('np_x_sg' + subgoal +'_no_imp.npy','np_y_sg' + subgoal +'_no_imp.npy', '../IGL_data/')
dataset2 = CustomDataSet('np_x_sg' + subgoal +'_imp.npy','np_y_sg' + subgoal +'_imp.npy', '../IGL_data/')
print(dataset2[0])
print(dataset2[1])
print(dataset2[2])
print(dataset2[3])
print(dataset2[4])
