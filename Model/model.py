import torch
import torch.nn as nn

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class IGL(nn.Module):
    def __init__(self, all_dim, device):
        super(IGL, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.net = nn.Sequential(
            nn.Linear(all_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.net_pos = nn.Sequential(
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 3)
        )
        self.net_grip = nn.Sequential(
        nn.Linear(256, 1),
        nn.Sigmoid()
        )

    def forward(self, state):
        # _input=torch.cat([state, stage],dim=1)
        common = self.net(state)
        pos = self.net_pos(common)
        grip = self.net_grip(common)
        output = torch.concat((pos,grip),1)
        return output


class hBC(nn.Module):
    def __init__(self, all_dim, device):
        super(hBC, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.net = nn.Sequential(
            nn.Linear(all_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.net_pos = nn.Sequential(
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 3)
        )
        self.net_grip = nn.Sequential(
        nn.Linear(256, 1),
        nn.Sigmoid()
        )

    def forward(self, state):
        common = self.net(state)
        pos = self.net_pos(common)
        grip = self.net_grip(common)
        output = torch.concat((pos,grip),1)
        return output

