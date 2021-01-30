# Only change model from train.py
import torch
import utils
import numpy as np
import torch.nn as nn
from autoencoder import AE2
from torch import optim
from torch.utils.data import DataLoader
#read in train data
trainX = np.load('trainX.npy')
img_dataset = utils.Image_Dataset(trainX)

utils.same_seeds(0)

model = AE2().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

model.train()
n_epoch = 30
# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
epoch_loss = 0
# 主要的訓練過程
for epoch in range(n_epoch):
    epoch_loss_prev = 100
    epoch_loss = 0
    for data in img_dataloader:
        img = data
        img = img.cuda()
        output1, output = model(img)
        loss = criterion(output, img)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        epoch_loss += loss.item()
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))
    if epoch_loss < epoch_loss_prev:
        torch.save(model.state_dict(),'checkpoints/last_checkpoint2.pth')
        print("---------------------------model saved----------------------------------")
    epoch_loss_prev = epoch_loss
