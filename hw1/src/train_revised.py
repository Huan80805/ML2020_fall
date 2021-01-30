import numpy as np
import glob
import pandas as pd
def read_traindata(filename):
    train_data = glob.glob(filename)
    print('training_dataset:',train_data)
    df = pd.concat([pd.read_csv(f) for f in train_data])
    preserved_columns = ['SO2','NOx','NO2','THC','PM10','PM2.5']
    df = df[preserved_columns]
    df = df.fillna('-')
    '''average = [2.22, 28.3, 19.9, 2.44, 46.80, 27.76]
    for i in range(len(preserved_columns)):
        df[df[preserved_columns[i]=='-']] = average[i]
        df[df[preserved_columns[i]=='nan']]= average[i]
    df[df=='-'] = 0
    df = df.fillna(0)'''
    df = df.reset_index(drop=True)
    data = np.array(df.values)
    print('rawdata_shape',data.shape)
    '''i = 0
    while i < data.shape[0]:
      try: 
        data[i]= data[i].astype(np.float64)
      except: 
        data = np.delete(data,np.s_[i-9:i+10],axis=0)
        i -= 10
      i += 1'''
    train_x = []
    train_y = []
    for i in range(data.shape[0]-9):
        x = data[i:i+9,:]
        y = data[i+9,5]
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
            if(data_x[i,2]>80 or data_x[i,2]< 0):
                return True
            if(data_x[i,3]>10 or data_x[i,3]<0):
                return True
            if(data_x[i,4]>150 or data_x[i,3]<0):
                return True
            if(data_x[i,5]>100 or data_x[i,4]<0):
                return True
        except:
            return True
        if (data_y<0 or data_y>100):
            return True
    return False

def RMSE(x , y , w , b):
    return np.sqrt(np.mean((np.dot(x,w)+b-y.reshape(-1,1))**2))

def training(train_x,train_y,validation_x,validation_y):
    # ref: https://reurl.cc/N6x4Rn
    #using sgd(batch),adam optimizer(m,beta,v,epislon),refularization(lamb)
    batch_size = 20
    total_epoch = 500
    data_len,dimension = train_x.shape[0],train_x.shape[1]
    w = np.full(dimension,0.1).reshape((-1,1))
    b = 0.1
    w_lr,b_lr  = 1e-4,1e-4
    lamb = 1e-3
    beta1,beta2 = 0.9, 0.999
    w_m = np.full(dimension,0).reshape((-1,1)).astype(np.float)
    w_v = np.full(dimension,0).reshape((-1,1)).astype(np.float)
    b_m = 0
    b_v = 0
    index = np.arange(data_len)
    epsilon = 1e-8
    t = 0
    for epoch in range (total_epoch):
        np.random.shuffle(index)
        train_x = train_x[index]
        train_y = train_y[index]
        for batch in range(data_len//batch_size):
            t += 1
            x_batch = train_x[batch*batch_size:(batch+1)*batch_size].reshape((batch_size,-1))
            y_batch = train_y[batch*batch_size:(batch+1)*batch_size].reshape((-1,1))
            loss = y_batch - np.dot(x_batch,w) - b
            w_g ,b_g = (-2*np.dot(x_batch.T,loss).astype(np.float)+2*lamb*np.sum(w)) ,(-2*loss.sum()) 
            w_m ,b_m = ((beta1*w_m)+(1-beta1)*w_g), ((b_m*beta1)+(1-beta1)*b_g)
            w_v ,b_v = ((beta2*w_v)+(1-beta2)*(w_g**2)) , ((beta2*b_v)+(1-beta2)*(b_g**2))
            w_m_hat ,b_m_hat = w_m/(1-beta1**t) ,b_m/(1-beta1**t)
            w_v_hat ,b_v_hat = w_v/(1-beta2**t) ,b_v/(1-beta2**t)
            w = w- (w_lr*w_m_hat/(epsilon+np.sqrt(w_v_hat)))
            b = b- (b_lr*b_m_hat/(epsilon+np.sqrt(b_v_hat)))
        print('epoch:',epoch)
        print('train RMSE:',RMSE(train_x,train_y,w,b))
        print('validation RMSE:',RMSE(validation_x,validation_y,w,b))
    return w,b
def save_model(w,b):
    df = pd.DataFrame(w)
    df = df.append([b],ignore_index = True)
    df.to_csv('../ML hw1/model.csv',header=None,index=None)
    
