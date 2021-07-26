from utils import *
from net import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob
import torch.nn as nn
import argparse

device = ('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100,
                    help='epoch number')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')

parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size')

parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer')               #'SGD' or 'Adam'
parser.add_argument('--backbone', type=str, default='vgg19',
                    help='vgg19,googlenet')

config = parser.parse_args(args=[])

def MPLN_optimizer(mtln):
    # 定义优化器
    if config.optimizer=='SGD':
        optimizer = torch.optim.SGD(mtln.parameters(), lr=config.lr)
    if config.optimizer == 'Adam':
        optimizer=torch.optim.Adam(mtln.parameters(),lr=config.lr)
    return optimizer


class my_dataset(Dataset):
    def __init__(self, store_path, split, name, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.transforms=data_transform
        self.names = name
        self.data_list = []
        self.label_list = []
        for n,i in enumerate(self.names):
            for file in glob.glob(self.store_path + '/' + split +'/'+i+ '/*.npy'):
                cur_path = file.replace('\\', '/')
                self.data_list.append(cur_path)
                self.label_list.append(n)

    def __getitem__(self, item):
        if self.transforms!=None:
            data = self.transforms(self.data_list[item])
        else:
            data=np.load(self.data_list[item])
        label = self.label_list[item]
        return data, label

    def __len__(self):
        return len(self.data_list)

def CE():
    # 定义损失函数
    loss=nn.CrossEntropyLoss()
    return loss

def train(inputs,targets,mtln,optimizer,train_accur,e):
    # 训练函数
    inputs=inputs.float().to(device)
    targets=targets.to(device)
    loss_sum=0
    output=torch.zeros((4,1,5))
    loss_ce = CE()
    for b in range(inputs.shape[0]):
        for imcoord in range(inputs.shape[1]):
            score = mtln(inputs[b][imcoord])
            if targets[b]=='4':
                loss_sum+=0.8*loss_ce(score,targets[b].unsqueeze(0))
            else:
                loss_sum += 0.4 * loss_ce(score, targets[b].unsqueeze(0))
            output[imcoord, :, :]=score
        pre_label = torch.argmax(torch.mean(output, dim=0))
        if targets[b] == pre_label:
            train_accur[e] += 1
    optimizer.zero_grad()
    loss_sum.backward(retain_graph=True)
    optimizer.step()
    return mtln,train_accur


if __name__ == '__main__':
    split = 'train'
    store_path = './'+config.backbone
    name=['000','001','002','003','004']
    train_dataset =my_dataset(store_path, split, name)

    mtln=MTLN().to(device)
    mtln.train()
    optimizer=MPLN_optimizer(mtln)
    train_accuracy = np.zeros(config.epochs)
    dataset_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    for e in range(config.epochs):
        for inputs,targets in dataset_loader:
            mtln,train_accuracy= train(inputs,targets,mtln, optimizer,train_accuracy,e)
        print('epoch:{},accuracy:{}'.format(e,train_accuracy[e]/500))
        if (e+1)%5==0:
            torch.save(mtln,'mtln{}.pth'.format(e+1))
            print('model saved!')
    train_accuracy/=500
    np.save('train_accuacy.npy',train_accuracy)