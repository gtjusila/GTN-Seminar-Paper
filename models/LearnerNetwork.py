import torch
import numpy
import torch.nn.functional as F
from torch import nn
class LearnerNetwork(nn.Module):
    def __init__(self, p,q,r):
        super(LearnerNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,p,kernel_size=3)
        nn.init.kaiming_normal_(self.conv1.weight, a = 0.1)
        self.batch_norm1 = nn.BatchNorm2d(p, momentum = 0.9) 
        
        self.conv2 = nn.Conv2d(p,q,kernel_size=3)
        nn.init.kaiming_normal_(self.conv2.weight, a = 0.1)
        self.batch_norm2 = nn.BatchNorm2d(q, momentum = 0.9)

        self.fully_connected = nn.Linear(q*4*4,r)
        nn.init.kaiming_normal_(self.fully_connected.weight, a = 0.1)
        self.batch_norm3 = nn.BatchNorm1d(r, momentum = 0.9)

        self.linear = nn.Linear(r,10)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,3,2)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x,negative_slope = 0.1)
        
        x = self.conv2(x)
        x = F.max_pool2d(x,3,2)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x,negative_slope = 0.1)
        
        x = torch.flatten(x,1)

        x = self.fully_connected(x)
        x = self.batch_norm3(x)
        x = F.leaky_relu(x,negative_slope = 0.1)

        x = self.linear(x)
        x = F.log_softmax(x,dim =1)
        return x