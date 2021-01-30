import sys
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = self.transform(img)
        return img
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
isGPU = torch.cuda.is_available()
print ('PyTorch GPU device is available: {}'.format(isGPU))
if isGPU is True:
    model.cuda()
model.eval()
img_path = "train/train"
test_set = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = dataset(test_set, transform)
test_loader = DataLoader(test_dataset, batch_size=1)
predict = []
for idx, img in enumerate(test_loader):
        if isGPU == True:
            img = img.cuda()
        with torch.no_grad():
            output = model(img)
            label_batch = torch.max(output, 1)[1].cpu().data.numpy()
            for label in label_batch:
                predict.append(label)
with open("train_predict.csv", 'w') as f:
    f.write('id,label\n')
    for column, test_label in  enumerate(predict):
        f.write('{},{}\n'.format(column, test_label))


