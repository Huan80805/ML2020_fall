## logistic
import numpy as np
import pandas as pd

x_df = pd.read_csv('.\X_train')
y_df = pd.read_csv('.\Y_train',header = None)
x_df = x_df.drop('fnlwgt',axis = 1)
x_df['age'] = (x_df['age']-1)//10
x_df['hours_per_week'] = x_df['hours_per_week']//10
cl_threshold=[3700,3000,2500,2000,1700,1400,1000,500,8]
cg_threshold=[75e3,50e3,25e3,10e3,7.5e3,5e3,2.5e3,1e3,8]
x_df['capital_gain'] = x_df['capital_gain']+9
x_df['capital_loss'] = x_df['capital_loss']+8
for i in range(len(cl_threshold)):
    x_df.loc[x_df['capital_gain']>cg_threshold[i],'capital_gain'] = i
for i in range(len(cl_threshold)):
    x_df.loc[x_df['capital_loss']>cl_threshold[i],'capital_loss'] = i
age = pd.get_dummies(x_df['age'], prefix='age')
capital_gain = pd.get_dummies(x_df['capital_gain'],prefix='capital_gain')
capital_loss = pd.get_dummies(x_df['capital_loss'],prefix='capital_loss')
hours_per_week =  pd.get_dummies(x_df['hours_per_week'], prefix='hours_per_week')
x_df = pd.concat([x_df.iloc[:,1],x_df.iloc[:,5:]],axis = 1)
x_df = pd.concat([age,capital_gain,capital_loss,hours_per_week,x_df],axis = 1)
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
def accuracy(x,y,w,b):
    correct = 0
    predict_y = sigmoid(np.dot(x,w)+b)
    for i in range(x.shape[0]):
        if (predict_y[i]>0.5 and y[i]==1) or(predict_y[i] <= 0.5 and y[i] == 0):
            correct += 1
    return correct/x.shape[0]
##logistic regression
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