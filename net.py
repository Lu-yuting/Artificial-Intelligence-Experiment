import numpy as np
import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F


def gray_into_CNN(X,Y,model):
    #CNN提取灰度图特征
    fea1=model(X)
    fea2=model(Y)
    return fea1,fea2

def TMP(x):
    #平均池化层
    x=x.squeeze(0)
    a=torch.zeros(512,14,14).to('cuda')
    torch.where(x>0,x,a)
    x=torch.mean(x,dim=1).unsqueeze(1)
    x=x.view(1,7168)
    return x

def ready_for_MTLN(X,Y,model):
    fea1,fea2=gray_into_CNN(X,Y,model)       #fea.size()=([1,512,14,14])通道数在前面
    t1=TMP(fea1)
    t2=TMP(fea2)
    T1=torch.cat((t1,t2),1)
    return T1

class MTLN(nn.Module):
    #多任务学习网络
    def __init__(self):
        super(MTLN,self).__init__()
        self.linear1=nn.Linear(2*7168,512)
        self.dropout = nn.Dropout(0.3)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(512,5)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        #x=x.view(1,2*7168)
        x=self.dropout(self.linear1(x))
        x=self.relu(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x


