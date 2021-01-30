from PIL import Image
import torch
import random
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
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
model.eval()
layer = model.conv4[0]
input_pic = "train/train/00000.jpg"
image = Image.open(input_pic)
transform = transforms.Compose([transforms.ToTensor()])
X = transform(image)
X = X.reshape(1,1,48,48)
Y = torch.tensor(1.).reshape(1,1,1,1).repeat(128,128,48,48)
X = X.repeat(128,128,1,1)
#output = layer(X)
output = layer(Y)
final = torch.tensor(0.).reshape(1,1).repeat(48,48)
for i in range(30):
    j = random.randint(0,127)
    k = random.randint(0,127)
    #plt.imshow(layer.weight[j,k].detach(),cmap='gray')
    #plt.imshow(output[j,k].detach(),cmap='gray')
    print(output[j,k])
    final += output[j,k]
    print(final)
final /= 30
print(final)
print(final.shape)
plt.imshow(final.detach(),cmap='gray')
plt.show()