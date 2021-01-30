#ref:https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
import os
from PIL import Image
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from random import randint
import matplotlib.pyplot as plt
import numpy as np

class naiveCNN(nn.Module):
    def __init__(self):
        super(naiveCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            )

        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 3*3*128)
        x = self.fc(x)
        return x

model = naiveCNN()
model.load_state_dict(torch.load('model.pkl'))
isGPU = False
#isGPU = torch.cuda.is_available()
print ('PyTorch GPU device is available: {}'.format(isGPU))
if isGPU is True:
    model.cuda()
model.eval()
for param in model.parameters():
    param.requires_grad = False
transform = transforms.Compose([transforms.ToTensor()])
def search_image(i):
    df = pd.read_csv("train/train/label.csv")
    label = df['label'].values
    index = np.where(label == i)[0]
    index = index[randint(0 , index.shape[0] - 1)]
    image_name = '{:0>5d}.jpg'.format(index)
    image_name = os.path.join("train/train", image_name)
    return image_name





for i in range(7):
    image_name =search_image(i)
    image = Image.open(image_name)
    image.save("sa_fig\{}.png".format(i))
    #image.show()
    X = transform(image)
    X = X.reshape((1,1,48,48))
    X.requires_grad_()
    score = model(X)
    score_max_index = score.argmax()
    score_max = score[0,score_max_index]
    score_max.backward()
    # 得到正确分类
    saliency, _ = torch.max(X.grad.data.abs(),dim=1)
    plt.imshow(saliency[0],cmap=plt.cm.gist_heat)
    plt.axis('off')
    plt.show()
    plt.savefig('sa_fig\sa{}'.format(i))


    

