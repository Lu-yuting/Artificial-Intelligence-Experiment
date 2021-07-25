import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

import argparse
device = ('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='vgg19',
                    help='backbone')               #'vgg19' or 'googlenet'
parser.add_argument('--split', type=str, default='test',
                    help='split')
parser.add_argument('--num', type=str, default='40',
                    help='model num')
config = parser.parse_args(args=[])

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



def test(inputs,targets,mtln,all,accuracy):
    batch=inputs.shape[0]
    inputs=inputs.float().to(device)
    targets=targets.numpy()
    output=torch.zeros((4,1,5))
    for b in range(batch):
        for imcoord in range(inputs.shape[1]):
            output[imcoord,:,:]=mtln(inputs[b][imcoord])
        pre_label=torch.argmax(torch.mean(output,dim=0))
        all[targets[b]] += 1
        #print(torch.mean(output,dim=0))
        #print(pre_label)
        if targets[b]==pre_label:
            accuracy[targets[b]]+=1
    return all,accuracy



if __name__ == '__main__':
    split = config.split
    store_path = './'+config.backbone
    name = ['000', '001', '002', '003', '004']
    model_path='./mtln'+config.num+'.pth'
    train_dataset = my_dataset(store_path, split, name)
    dataset_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    all=[0,0,0,0,0]
    accuracy=[0,0,0,0,0]
    mtln = torch.load(model_path).to(device)
    mtln.eval()
    for n,(inputs,targets) in enumerate(dataset_loader):
        all,accuracy = test(inputs,targets, mtln,all,accuracy)
    print('总准确率是:{}'.format(np.sum(accuracy)/np.sum(all)))
    print('每一类的准确率是:{} {} {} {} {}'.format(accuracy[0]/all[0],accuracy[1]/all[1],accuracy[2]/all[2],accuracy[3]/all[3],accuracy[4]/all[4]))