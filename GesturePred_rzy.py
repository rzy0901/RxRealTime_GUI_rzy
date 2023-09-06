from PMR_fft import PMR_fft

import numpy as np
from PIL import Image
import subprocess
import torch 
from torch import nn
import torch.nn.functional as nf
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18


# For model 1 (pure simu-to-real inference), 
# run https://github.com/Jcq242818/mediapipe_spectrogram_classification/blob/change_model_order/jcq_train.py for detailed normMean and normStd.
normMean = [0.026820642701525108, 0.047705834694988805, 0.5362284994553294]
normStd = [0.08753696827923, 0.13767478809434114, 0.0832230032303204]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform,
])

class fc_part(nn.Module):
        # fc 全连接层
        def __init__(self):
            super().__init__()
            # self.fc1 = nn.Linear(512,512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 120)
            self.fc3 = nn.Linear(120,3)

        def forward(self, x):
            x = nf.relu(self.fc1(x))
            x = nf.relu(self.fc2(x))
            x = nf.relu(self.fc3(x))
            # x = self.fc1(x)
            return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ModelInit():
    model = resnet18(pretrained=True).to(device)
    model.fc = fc_part().to(device)
    model_Path = './resnet18_model.pth'
    model.load_state_dict(torch.load(model_Path))
    return model

def GesPre(img:torch.Tensor,model):
    print(img.shape)
    img = img.unsqueeze(0)
    with torch.no_grad():
        inputs = img.to(device)
        outputs = model.forward(inputs)
        _,predicted = torch.max(outputs,axis = 1)
    return predicted
        
def main():
    filename = './data1.dat'
    data = np.fromfile(filename,dtype = '<f', count = -1,).reshape(2,-1,order = "F")
    data_complex = data[0,:] + 1j*data[1,:]
    data_sample = data_complex.reshape(-1,2,order = "F")
    data_sample = np.transpose(data_sample)
    # Python version
    # PMR_fft(data_sample,noAxes=True)
    # Matlab version
    matlab_command = "matlab -nodisplay -nosplash -r \"func_PMR_fft('{}','./temp_matlab.jpg','./temp_resize_matlab.jpg'); exit;\"".format(filename)
    print(matlab_command)
    try:
        subprocess.run(matlab_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    img = Image.open('./temp_resize_matlab.jpg').convert('RGB')
    print(img)
    # defualt display
    img.convert('RGB').show(title="Spectrogram")
    img = testTransform(img)
    model = ModelInit()
    predicted = GesPre(img,model)
    print("Class",int(predicted))
if __name__ == '__main__':
    main()