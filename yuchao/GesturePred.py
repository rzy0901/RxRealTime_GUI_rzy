import torch 
from torch import nn,optim, tensor
import sys
sys.path.append("..")
import numpy as np
from matplotlib import pyplot as plt
import time
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='ignore')

import glob #返回文件路径中所有匹配的文件名

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18

def transformsNorm(data):
    data = (data  - data.mean())/data.std()
    return data

class fc_part(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,120)
        self.fc4 = nn.Linear(120,6)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class MyDataset(Dataset):
    def __init__(self,FileName):
        imgs =[FileName]
        self.imgs = imgs 

    def __getitem__(self,index):
        fn = self.imgs[index]

        img = torch.tensor(np.load(fn))
        img = img.type(torch.FloatTensor)
        img = torch.unsqueeze(img, dim=0)
        return img,0
    def __len__(self):
        return len(self.imgs)

class RealTimeData(Dataset):
    def __init__(self,data):
        self.img = torch.tensor(data) 

    def __getitem__(self,index):
        fn = self.imgs[index]
        img = torch.tensor(np.load(fn))
        img = img.type(torch.FloatTensor)
        img = torch.unsqueeze(img, dim=0)
        return img,0

def DataInit(data):
        data = transformsNorm(data)
        img = torch.tensor(data)
        print(sum(img[5,:]))
        print(type(img),img.shape)
        img = img.type(torch.FloatTensor)
        img = torch.unsqueeze(img, dim=0)
        img = torch.unsqueeze(img, dim=0)
        return img

def predict_gesture(model,test_loader,device):
    with torch.no_grad():
        for data,_ in test_loader :            
            model.eval()
            inputs = data
            inputs = inputs.to(device)
            outputs = model.forward(inputs)
            _,predicted = torch.max(outputs,axis = 1)
    return predicted

def show_real_v2(A_TD,Pred):
    global fs, CIT, N_slide, T_slide
    fs = 10e6
    CIT = 0.1
    N_slide = 10
    T_slide = CIT / N_slide
    total_duration = 2
    print(total_duration)
    array_start_time =np.arange(0,total_duration-CIT+T_slide,T_slide)
    Doppler_frequency = 600
    step_dop = 1/CIT
    array_Doppler_frequency = np.arange(-Doppler_frequency-1,Doppler_frequency,step_dop)
    pause_time = 5
    x, y = np.meshgrid(array_start_time,array_Doppler_frequency)
    plot_A_TD = np.zeros(np.size(A_TD))
    plt.figure(1)
    plt.ion()
    # plot_A_TD = np.transpose(20*(np.log(abs(A_TD)/np.max((np.max(abs(A_TD)))))/np.log(20)))
    plot_A_TD = np.transpose(A_TD)
    plt.clf()
    plt.pcolor(x,y,plot_A_TD.T,shading='auto',cmap='jet',vmin=-25,vmax=-5)        
    plt.colorbar()
    #vmin=-15,vmax=0
    plt.ylabel('Doppler frequency (Hz)')
    plt.xlabel('time (s)')
    plt.title('Gesture ' + str(int(Pred)))
    # plt.show()
    plt.pause(pause_time)
    plt.ioff()

#模型提取
def ModelInit():
    # device = 'cpu'
    model = resnet18(pretrained = True)
    model.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = fc_part()
    # model_Path = '/home/yuc/DL/Deep_Learning/PMR_0114_npyNorm/resnet18_model_TransForm.pt'
    model_Path = '/media/rzy/76800D98800D5FCB/Codes/RxRealTime_GUI_rzy/yuchao/resnet18_model_TransForm.pt'
    new_model = torch.load(model_Path, map_location='cpu')
    model.load_state_dict(new_model)
    # model = model.to(device)
    return model

def GesPre(data,model):
    print(data.shape)
    with torch.no_grad():
        Input = DataInit(data)
        model.eval()
        outputs = model.forward(Input)
        _,predicted = torch.max(outputs,axis = 1)
    return predicted

def main():
    FileName = '/home/yuc/DL/Deep_Learning/PMR_0114_npyNorm/All/Ges2/NormData_npy121.npy'

    model = ModelInit()
    #数据读取
    data = np.load(FileName)

    predicted = GesPre(data,model)
    print("Class :",int(predicted))
    show_real_v2(data,predicted)
    

if __name__ == '__main__':
    main()