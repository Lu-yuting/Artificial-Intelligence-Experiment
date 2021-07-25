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

parser.add_argument('--backbone', type=str, default='vgg19',
                    help='backbone')               #'vgg19' or 'googlenet'



parser.add_argument('--split', type=str, default='train',
                    help='split')

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
            if split =='train':
                for file in glob.glob(self.store_path + '/' + split +'/'+i+ '/*.npy'):
                    cur_path = file.replace('\\', '/')
                    if n==4:
                        for j in range(10):
                            self.data_list.append(cur_path)
                            self.label_list.append(n)
                    else:
                        self.data_list.append(cur_path)
                        self.label_list.append(n)
            else:
                for file in glob.glob(self.store_path + '/' + split +'/'+i+ '/*.npy'):
                    cur_path = file.replace('\\', '/')
                    self.data_list.append(cur_path)
                    self.label_list.append(n)


    def __getitem__(self, item):
        data = self.transforms(self.data_list[item])
        label = self.label_list[item]
        return data, label

    def __len__(self):
        return len(self.data_list)

def save_feature(feature,target,n):
    target=target.cpu().numpy()
    np.save(os.getcwd()+'/'+config.backbone+'/'+config.split+'/'+'00'+str(target)+'/'+'feature'+str(n)+'_'+str(target)+'.npy',feature)



def pre_function(inputs,targets,model,n):
    inputs=inputs.to(device)
    targets=targets.to(device)
    inputs=inputs.squeeze(1)

    for b in range(inputs.shape[0]):
        feature=np.zeros((4,1,14336))
        for i in range(inputs.shape[1]):
            inputs1=inputs[b, i, :, :, 0].unsqueeze(0)
            inputs2=inputs[b, i, :, :, 1].unsqueeze(0)
            input1=torch.from_numpy(np.zeros((1,3,224,224)).astype(np.float32)).to(device)
            input1[0,0, :, :] = inputs1
            input1[0,1, :, :] = inputs1
            input1[0,2, :, :] = inputs1
            input2 = torch.from_numpy(np.zeros((1,3, 224, 224)).astype(np.float32)).to(device)
            input2[0,0,:,:]=inputs2
            input2[0,1, :, :] = inputs2
            input2[0,2, :, :] = inputs2
            T=ready_for_MTLN(input1,input2,model)
            feature[i]=T.detach().cpu().numpy()
        save_feature(feature,targets[b],n)
        n+=1
    return n



if __name__ == '__main__':
    split = config.split
    store_path = './data'
    name=['000','001','002','003','004']
    if split=='train':
        train_dataset =my_dataset(store_path, split, name, pre_treat_train)
    else:
        train_dataset = my_dataset(store_path, split, name, pre_treat_test)
    # 下面加载所需网络的预训练参数
    if config.backbone=='vgg19':

        model_before=models.vgg19(pretrained=True)
        t=[]
        for i,layer in enumerate(model_before.children()):
            for j,sublayer in enumerate(layer.children()):
                 if i==0 and j<30:
                    t.append(sublayer)
        model = nn.Sequential(*t)

    else:
        model_before = models.googlenet(pretrained=True)
        model = nn.Sequential(*list(model_before.children())[0:11])

    model = model.to(device)
    n=0
    dataset_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for inputs,targets in dataset_loader:
        n=pre_function(inputs,targets, model,n)