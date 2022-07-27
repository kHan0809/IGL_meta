import torch
import torch.nn as nn
from Model.model import hBC, InvKin
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import deque

class Buffer:
    def __init__(self,o_dim,a_dim,buffer_size = 1000000):
        self.size = buffer_size
        self.num_experience = 0
        self.o_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.a_mem = np.empty((self.size, a_dim), dtype=np.float32)
        self.no_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.r_mem = np.empty((self.size, 1), dtype=np.float32)
        self.done_mem = np.empty((self.size, 1), dtype=np.float32)
    def store_sample(self,o,a,r,no,done):
        idx = self.num_experience%self.size
        self.o_mem[idx] = o
        self.a_mem[idx] = a
        self.r_mem[idx] = r
        self.no_mem[idx] = no
        self.done_mem[idx] = done
        self.num_experience += 1
    def random_batch(self, batch_size = 256):
        N = min(self.num_experience, self.size)
        idx = np.random.choice(N,batch_size)
        o_batch = self.o_mem[idx]
        a_batch = self.a_mem[idx]
        r_batch = self.r_mem[idx]
        no_batch = self.no_mem[idx]
        done_batch = self.done_mem[idx]
        return o_batch, a_batch, r_batch, no_batch, done_batch
    def all_batch(self):
        N = min(self.num_experience,self.size)
        return self.o_mem[:N], self.a_mem[:N], self.r_mem[:N], self.no_mem[:N], self.done_mem[:N]
    def store_demo(self,demo):
        demo_len= len(demo)-1
        self.o_mem[:demo_len]  = demo[:-1]
        self.no_mem[:demo_len] = demo[1:]
        self.num_experience += demo_len





class BCO(nn.Module):
    def __init__(self, state_dim,action_dim,expert_demo, args):
        super(BCO, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.BC = hBC(state_dim,args.device_train).to(self.args.device_train)
        self.InvKin = InvKin(state_dim,args.device_train).to(self.args.device_train)
        self.Inv_buffer  = Buffer(state_dim, action_dim)
        self.Demo_buffer = Buffer(state_dim, action_dim)
        self.Demo_buffer.store_demo(expert_demo)

        self.InvKin_optim   = torch.optim.Adam(self.InvKin.parameters(),lr=args.BCO_lr,weight_decay=args.BCO_wd)

    def BC_train(self):
        self.BC.train()
        o,a,r,no,done = self.Demo_buffer.all_batch()
        x = np.concatenate((o,no),1)
        train_size = int(len(x) * 0.7)
        with torch.no_grad():
            action = self.InvKin(torch.FloatTensor(x).to(self.args.device_train)).cpu().detach().numpy()
        idx = np.random.choice(len(x), train_size)
        train_x, train_y = x[idx], action[idx]
        iidx = np.delete(np.array(list(range(len(x)))), idx)
        valid_x, valid_y = x[iidx], action[iidx]

        train_batch = (len(idx)  // 5) - 2
        valid_batch = (len(iidx) // 5) - 2

        train_dataset = CustomDataSet(train_x, train_y)
        valid_dataset = CustomDataSet(valid_x, valid_y)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch)
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=valid_batch)

        mse_loss = nn.MSELoss()
        total_loss = 0
        valid_patience = deque(maxlen=5)
        for i in range(self.args.BCO_BC_epoch):
            for j, (input_,label_) in enumerate(train_loader):
                self.InvKin_optim.zero_grad()
                output = self.InvKin(input_.type(torch.FloatTensor).to(self.args.device_train))
                loss   = mse_loss(label_.type(torch.FloatTensor).to(self.args.device_train),output)
                loss.backward()
                self.InvKin_optim.step()
                total_loss += loss.item()

            valid_loss = 0
            for j, (input_, label_) in enumerate(valid_loader):
                with torch.no_grad():
                    output = self.InvKin(input_.type(torch.FloatTensor).to(self.args.device_train))
                    loss = mse_loss(label_.type(torch.FloatTensor).to(self.args.device_train), output)
                valid_loss += loss.item()
            valid_patience.append(valid_loss)

            valid_flag = True
            for k in range(len(valid_patience)-1):
                if valid_patience[k] > valid_patience[k+1]:
                    valid_flag = False
            if len(valid_patience)>4 and valid_flag:
                print("=======Early Stop!========")
                break
        print("[BC train epoch] :", i + 1, "[loss] : ", total_loss)




    def Inv_train(self):
        self.InvKin.train()
        o,a,r,no,done = self.Inv_buffer.all_batch()
        x = np.concatenate((o,no),1)
        train_size = int(len(x)*0.7)
        idx = np.random.choice(len(x), train_size)
        train_x,train_y = x[idx], a[idx]
        iidx = np.delete(np.array(list(range(len(x)))),idx)
        valid_x,valid_y = x[iidx], a[iidx]

        train_batch = (len(idx) // 5) - 2
        valid_batch = (len(iidx) // 5) - 2

        train_dataset = CustomDataSet(train_x, train_y)
        valid_dataset = CustomDataSet(valid_x, valid_y)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch)
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=valid_batch)

        mse_loss = nn.MSELoss()
        total_loss = 0
        valid_patience = deque(maxlen=5)
        for i in range(self.args.BCO_InvKin_epoch):
            for j, (input_,label_) in enumerate(train_loader):
                self.InvKin_optim.zero_grad()
                output = self.InvKin(input_.type(torch.FloatTensor).to(self.args.device_train))
                loss   = mse_loss(label_.type(torch.FloatTensor).to(self.args.device_train),output)
                loss.backward()
                self.InvKin_optim.step()
                total_loss += loss.item()

            valid_loss = 0
            for j, (input_, label_) in enumerate(valid_loader):
                with torch.no_grad():
                    output = self.InvKin(input_.type(torch.FloatTensor).to(self.args.device_train))
                    loss = mse_loss(label_.type(torch.FloatTensor).to(self.args.device_train), output)
                valid_loss += loss.item()
            valid_patience.append(valid_loss)

            valid_flag = True
            for k in range(len(valid_patience)-1):
                if valid_patience[k] > valid_patience[k+1]:
                    valid_flag = False
            if len(valid_patience)>4 and valid_flag:
                print("=======Early Stop!========")
                break
        print("=================================================")
        print("[Inv train epoch] :",i+1,"[loss] : ",total_loss)
        self.Inv_buffer.__init__(self.state_dim,self.action_dim)










class CustomDataSet(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
