import numpy as np
import pandas as pd
import glob
def read_traindata(filename):
    train_data = glob.glob(filename)
    print('training_dataset:',train_data)
    df = pd.concat([pd.read_csv(f) for f in train_data])
    preserved_columns = ['SO2','NOx','NO2','PM10','PM2.5']
    df = df[preserved_columns]
    df = df.fillna('-')
    df = df.reset_index(drop=True)
    data = np.array(df.values)
    print('rawdata_shape',data.shape)
    train_x = []
    train_y = []
    for i in range(data.shape[0]-9):
        x = data[i:i+9,:]
        y = data[i+9,4]
        if not abnormal_data(x,y):
            train_x.append(x.reshape(-1))
            train_y.append(y)
    train_x = np.array(train_x).astype(np.float)
    train_y = np.array(train_y).astype(np.float)
    data_len = train_x.shape[0]//2
    print('data_len:',data_len)
    validation_x = train_x[data_len:,:]
    validation_y = train_y[data_len:]
    train_x = train_x[:data_len,:]
    train_y = train_y[:data_len]
    return(train_x,train_y,validation_x,validation_y)
def abnormal_data(data_x,data_y):
    for i in range(9):
        try:
            data_x[i] = data_x[i].astype(np.float)
            data_y = float(data_y)
            if(data_x[i,0]>10 or data_x[i,0]< 0):
                return True
            if(data_x[i,1]>100 or data_x[i,1]< 0):
                return True
            if(data_x[i,2]>50 or data_x[i,2]< 0):
                return True
            if(data_x[i,3]>150 or data_x[i,3]<0):
                return True
            if(data_x[i,4]>100 or data_x[i,4]<0):
                return True
        except:
            return True
        if (data_y<0 or data_y>79):
            return True
    return False

def training(train_x,train_y,validation_x,validation_y):
    #using adagrad only
    batch_size = 5
    total_epoch = 1000
    data_len,dimension = train_x.shape[0],train_x.shape[1]
    w = np.full(dimension,0.1).reshape((-1,1)).astype(np.float)
    b = 0.1
    lamb = 1e-3
    epsilon = 1e-8
    lr = 1e-2
    adagrad_w = np.full(dimension,0.).reshape((-1,1))
    adagrad_b = 0
    for epoch in range (total_epoch):
        for batch in range(data_len//batch_size):
            x_batch = train_x[batch*batch_size:(batch+1)*batch_size].reshape((batch_size,-1))
            y_batch = train_y[batch*batch_size:(batch+1)*batch_size].reshape((-1,1))
            loss = y_batch - np.dot(x_batch,w) - b
            grad_w , grad_b = (-2*np.dot(x_batch.T,loss)+2*lamb*np.sum(w)) ,(-2*loss.sum())
            adagrad_w += (grad_w)**2
            adagrad_b += (grad_b)**2
            w -= (lr*grad_w/(adagrad_w+epsilon)**(1/2))
            b -= (lr*grad_b/(adagrad_b+epsilon)**(1/2))
        print('epoch:',epoch)
        print('train RMSE:',np.sqrt(np.mean((np.dot(train_x,w)+b-train_y.reshape(-1,1))**2)))
    print('validation RMSE:',np.sqrt(np.mean((np.dot(validation_x,w)+b-validation_y.reshape(-1,1))**2)))
    return w,b
def save_model(w,b):
    df = pd.DataFrame(w)
    df = df.append([b],ignore_index = True)
    df.to_csv('../ML hw1/model.csv',header=None,index=None)

train_x,train_y,validation_x,validation_y =read_traindata('./train_datas_*.csv')      
w,b = training(train_x,train_y,validation_x,validation_y)
save_model(w,b)

