##logistic
import numpy as np
import pandas as pd

x_df = pd.read_csv('.\X_train')
y_df = pd.read_csv('.\Y_train',header = None)
x_df = x_df.drop('fnlwgt',axis = 1)
train_x = np.array(x_df.values).astype(np.float)
train_y = np.array(y_df.values).astype(np.int)

dimension = train_x.shape[1]
data_len = train_x.shape[0]//2
validation_x = train_x[data_len:]
validation_y = train_y[data_len:]
train_x = train_x[:data_len]
train_y = train_y[:data_len]
def sigmoid(z):
    return np.clip((1/(1+np.exp(-z))),1e-10,1-1e-10)
def normalize(train_x,validation_x):
    columns_to_normalized = [0,2,3,4]
    mean = train_x[:,columns_to_normalized].mean(axis=0)
    std = train_x[:,columns_to_normalized].std(axis=0)
    train_x[:,columns_to_normalized] = (train_x[:,columns_to_normalized]-mean)/std
    validation_x[:,columns_to_normalized] = (validation_x[:,columns_to_normalized]-mean)/std
    return train_x,validation_x,mean,std
    pass
def accuracy(x,y,w,b):
    correct = 0
    predict_y = sigmoid(np.dot(x,w)+b)
    for i in range(x.shape[0]):
        if (predict_y[i]>0.5 and y[i]==1) or(predict_y[i] <= 0.5 and y[i] == 0):
            correct += 1
    return correct/x.shape[0]
train_x,validation_x,mean,std = normalize(train_x,validation_x)
total_epoch = 300
w = np.full(dimension,0.).reshape((-1,1))
b = 0.1
adagrad_w = np.full(dimension,0.).reshape((-1,1))
adagrad_b = 0.
epsilon = 1e-8
lamb = 0
lr = 0.05
batch_size = 5
for epoch in range(total_epoch):
    for batch in range(data_len//batch_size):
        x = train_x[batch*batch_size:(batch+1)*batch_size].reshape((batch_size,-1))
        y = train_y[batch*batch_size:(batch+1)*batch_size].reshape((batch_size,-1))
        predict_y = sigmoid(np.dot(x,w)+b)
        loss = y - predict_y
        grad_w = -1*np.dot(x.T,loss)
        grad_b = -1*loss.sum()
        adagrad_w += (grad_w)**2
        adagrad_b += (grad_b)**2
        w -= lr/np.sqrt(adagrad_w+epsilon)*grad_w
        b -= lr/np.sqrt(adagrad_b+epsilon)*grad_b
        loss = -1*np.mean(y*np.log(predict_y)+(1-y)*np.log(1-predict_y))
    print('loss:',loss)
    print('epoch:',epoch)
    print('train accuracy:',accuracy(train_x,train_y,w,b))
    print('validation accuracy:',accuracy(validation_x,validation_y,w,b))

np.save('model.npy',[w,b])
np.save('parameter.npy',[mean,std])
