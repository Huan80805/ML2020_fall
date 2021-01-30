#Using dictionary to create BOW
import gc
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset , DataLoader
from DNN_model import DNN
import torch.nn as nn
from torch.optim import Adam

def token_extract(token):
    return not(token.is_punct or token.is_stop)

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
    outputs[outputs<0.5] = 0 # 小於 0.5 為負面
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class dataset(Dataset):
	def __init__(self , data , label):
		self.data = data
		self.label = label
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return (torch.FloatTensor(self.data[index]) , self.label[index])

def load_data():
    NLP = spacy.load('en_core_web_sm')
    dictionary = np.load('dictionary.npy')
    with open('training_label.txt', 'r',encoding="utf-8") as f:
        data = f.readlines()
        train_y = [line[0] for line in data]
        data = [' '.join(line.strip('\n').split(' ')[2:]) for line in data]
        print('-----------loading training data---------------')
        train_x = []
        for i,line in enumerate(data):
            BOW = np.zeros(dictionary.shape[0])
            for token in NLP(line):
                if token_extract(token):
                    index = np.where(dictionary == token.text)[0][0]
                    BOW[index] = 1
            print('sentence count #{}'.format(i+1), end='\r')
            train_x.append(BOW)
            del BOW
            del line
            gc.collect()
        train_x = np.array(train_x)    
        validation_x = train_x[180000:]
        validation_y = train_y[180000:]
        train_x = train_x[:180000 ]
        train_y = train_y[:180000]
        print("\n-----------training data loaded---------------")
    return (train_x , train_y , validation_x , validation_y)

def train(batch_size,epochs,lr,train_loader,validation_loader,model , device):    
    model.train()
    model.to(device)
    criterion = nn.BCELoss()
    t_batch = len(train_loader) 
    v_batch = len(validation_loader) 
    optimizer = Adam(model.parameters() , lr = lr)
    total_loss,total_acc,best_acc = 0,0,0
    for epoch in range(epochs):
        total_loss,total_acc = 0
        for i , (data , label) in enumerate(train_loader):
            data , label = data.to(device) , label.to(device)
            optimizer.zero_grad()
            y = model(data)
            y = y.squeeze()
            loss = criterion(y,label)
            loss.backward()
            optimizer.step()
            correct = evaluation(y,label)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        model.eval()
        with torch.no_grad():
            total_loss,total_acc = 0,0
            for i, (data , label) in enumerate(validation_loader):
                data = data.to(device)
                label = label.to(device)
                y = model(data)
                y = y.squeeze()
                loss = criterion(y,label)
                correct = evaluation(y, label) 
                total_acc += (correct / batch_size)
                total_loss += loss.item()
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "ckpt.model")
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() 

if (__name__ == '__main__'):
    batch_size = 32
    epoch = 5
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train_x , train_y , validation_x , validation_y) = load_data()
    train_dataset = dataset(train_x , train_y)
    validation_dataset = dataset(validation_x , validation_y)
    train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True)
    validation_loader = DataLoader(validation_dataset , batch_size = batch_size , shuffle = False)
    model = DNN(train_x.shape[1])
    train(batch_size,epoch,lr,train_loader, validation_loader ,model , device)

