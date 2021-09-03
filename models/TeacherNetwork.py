import torch
import numpy
import torch.nn.functional as F
from torch import nn

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork,self).__init__()
        self.fc_1 = nn.Linear(138,1024)
        nn.init.kaiming_normal_(self.fc_1.weight, a = 0.1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc_2 = nn.Linear(1024,6272)
        nn.init.kaiming_normal_(self.fc_2.weight, a = 0.1)
        self.bn2 = nn.BatchNorm1d(6272)
        self.conv1 = nn.Conv2d(128, 64, kernel_size = 3,padding=1,padding_mode='reflect')
        nn.init.kaiming_normal_(self.conv1.weight, a = 0.1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size = 3,padding=1,padding_mode='reflect')
        nn.init.kaiming_normal_(self.conv2.weight, a = 0.1)
        self.tanh = nn.Tanh()
        self.learner_optim_params = nn.Parameter(torch.tensor([0.02, 0.5]), True)
        self.learner_input = nn.Parameter(torch.randn(32,128,128),True)
    def forward(self,x):
        x = self.fc_1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.1)

        x = self.fc_2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.1)

        x = x.view(128,128,7,7)
        x = F.interpolate(x,scale_factor = 2)
        x = self.conv1(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = F.interpolate(x,scale_factor = 2)
        x = self.conv2(x)
        x = self.tanh(x)
        
        return x
