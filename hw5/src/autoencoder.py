import torch.nn as nn
from torchsummary import summary
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,return_indices=True)
        )
        self.unpool1 = nn.MaxUnpool2d(2)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,return_indices=True)
        )
        self.unpool2 = nn.MaxUnpool2d(2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,return_indices=True)
        )
        self.unpool3 = nn.MaxUnpool2d(2)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            #nn.MaxPool2d(2)
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
 
    def forward(self, x):
        x_en, indices1 = self.encoder1(x)
        x_en, indices2 = self.encoder2(x_en)
        x_en, indices3 = self.encoder3(x_en)
        x_en = self.encoder4(x_en)
        x = self.decoder4(x_en)
        x = self.unpool3(x, indices3)
        x = self.decoder3(x)
        x = self.unpool2(x, indices2)
        x = self.decoder2(x)
        x = self.unpool1(x, indices1)
        x = self.decoder1(x)
        
        return x_en,x

class AE2(nn.Module):
    def __init__(self):
        super(AE2, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.maxpool1=nn.MaxPool2d(2,return_indices=True)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.decoder1 = nn.Sequential(
             nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
             nn.Tanh()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.maxpool2 = nn.MaxPool2d(2,return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.maxpool3 = nn.MaxPool2d(2,return_indices=True)
        self.unpool3 = nn.MaxUnpool2d(2)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
    def forward(self, x):
        x_en = self.encoder1(x)
        size1 = x_en.size()
        x_en, indices1 = self.maxpool1(x_en)
        x_en = self.encoder2(x_en)
        size2 = x_en.size()
        x_en, indices2 = self.maxpool2(x_en)
        x_en = self.encoder3(x_en)
        size3 = x_en.size()
        x_en, indices3 = self.maxpool3(x_en)
        x_en = self.encoder4(x_en)
        x = self.decoder4(x_en)
        x = self.unpool3(x,indices=indices3,output_size=size3)
        x = self.decoder3(x)
        x = self.unpool2(x,indices=indices2,output_size=size2)
        x = self.decoder2(x)
        x = self.unpool1(x,indices=indices1,output_size=size1)
        x = self.decoder1(x)        
        return x_en,x

class AE3(nn.Module):
    def __init__(self):
        super(AE3, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,return_indices=True)
        )
        self.unpool1 = nn.MaxUnpool2d(2)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,return_indices=True)
        )
        self.unpool2 = nn.MaxUnpool2d(2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
 
    def forward(self, x):
        x_en, indices1 = self.encoder1(x)
        x_en, indices2 = self.encoder2(x_en)
        x_en  = self.encoder3(x_en)
        x = self.decoder3(x_en)
        x = self.unpool2(x, indices2)
        x = self.decoder2(x)
        x = self.unpool1(x, indices1)
        x = self.decoder1(x)
        
        return x_en,x

if __name__ == "__main__":
    ae = AE().cuda()
    summary(ae, (3, 32, 32))
    ae3 = AE3().cuda()
    summary(ae3,(3,32,32))