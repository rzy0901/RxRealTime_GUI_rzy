from PMR_fft import PMR_fft

import numpy as np
from PIL import Image
import subprocess
import torch
from torch import nn
import torch.nn.functional as nf
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import matlab.engine
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For model 1 (pure simu-to-real inference),
# run https://github.com/Jcq242818/mediapipe_spectrogram_classification/blob/change_model_order/jcq_train.py for detailed normMean and normStd.
normMean = [0.04035673338779959, 0.06492115332243965, 0.5282050517429131]
normStd = [0.12570609443609027, 0.1726390070527584, 0.08701469187852846]
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
        self.fc3 = nn.Linear(120, 3)

    def forward(self, x):
        x = nf.relu(self.fc1(x))
        x = nf.relu(self.fc2(x))
        x = nf.relu(self.fc3(x))
        # x = self.fc1(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def ModelInit():
    model = resnet18(pretrained=True).to(device)
    # model = model.to(device)
    model.fc = fc_part().to(device)
    model_Path = './best_resnet18_model.pth'
    model.load_state_dict(torch.load(model_Path,map_location='cpu'))
    return model


def GesPre(img: torch.Tensor, model):
    print(img.shape)
    img = img.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        inputs = img.to(device)
        outputs = model.forward(inputs)
        _, predicted = torch.max(outputs, axis=1)
    return predicted


def data_save(data, filename):
    with open(filename, 'wb') as fid:
        data.tofile(fid)


def PMR_fft_matlab(data, imshow=False):
    # generate temp_matlab.jpg and temp_resize_matlab.jpg by matlab
    # return numpy img
    data_save(data, './temp.dat')
    matlab_command = "matlab -nodisplay -nosplash -r \"func_PMR_fft('./temp.dat','./temp_matlab.jpg','./temp_resize_matlab.jpg',false); exit;\""
    print(matlab_command)
    try:
        subprocess.run(matlab_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    img = Image.open('./temp_resize_matlab.jpg').convert('RGB')
    if imshow:
        Image.open('./temp_matlab.jpg').convert('RGB').show(title="Spectrogram")
    return img

def PMR_fft_matlabEngine(eng, data, imshow=False):
    data_save(data, './temp.dat')
    print('temp.dat saved!')
    try:
        eng.func_PMR_fft('./temp.dat', './temp_matlab.jpg',
                         './temp_resize_matlab.jpg', imshow, nargout=0)
    except Exception as e:
        print(f"Error: {e}")
    img = Image.open('./temp_resize_matlab.jpg').convert('RGB')
    return img

def dat2Npy(data_path):
    data = np.fromfile(data_path, dtype='<f', count=-1,).reshape(2, -1, order="F")
    data_complex = data[0, :] + 1j*data[1, :]
    data_sample = data_complex.reshape(-1, 2, order="F")
    data_sample = np.transpose(data_sample)
    return data_sample

def main():
    data_sample = dat2Npy("I:\\实验\\data_0112\\data261.dat")
    # data_sample = np.load('./data1.npy')
    # # Python version for faster speed
    # img = PMR_fft(data_sample, noAxes=True,imshow=True)
    # Matlab version for higher recognition accuracy
    eng = matlab.engine.start_matlab()
    print("Matlab engine start")
    # img = PMR_fft_matlab(data_sample,imshow=True)
    img = PMR_fft_matlabEngine(eng, data_sample, imshow=True)
    # Convet to tensor
    img = testTransform(img)
    model = ModelInit()
    predicted = GesPre(img, model)
    print("Class", int(predicted))
    input('Press any key to continue.')
    eng.quit()

if __name__ == '__main__':
    main()
