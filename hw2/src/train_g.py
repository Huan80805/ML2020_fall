##generative
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
def cov(Class,mean):
    cov = np.full((Class.shape[1],Class.shape[1]),0.)
    for i in range(Class.shape[0]):
        cov += np.dot((Class[i]-mean).T,Class[i]-mean)
    cov /= Class.shape[0]
    return cov
train_x,validation_x,mean,std = normalize(train_x,validation_x)
c0 = train_x[np.repeat(train_y,train_x.shape[1],axis=1) == 0].reshape((-1,train_x.shape[1]))
c1 = train_x[np.repeat(train_y,train_x.shape[1],axis=1) == 1].reshape((-1,train_x.shape[1]))
p_c0 = c0.shape[0]/train_x.shape[0]
p_c1 = c1.shape[0]/train_x.shape[0]
mean_c0 = np.mean(c0,axis=0).reshape((1,-1))
mean_c1 = np.mean(c1,axis=0).reshape((1,-1))
cov_0 = cov(c0,mean_c0)+1e-9
cov_1 = cov(c1,mean_c1)
cov = p_c0*cov_0 + p_c1*cov_1
cov_inv = np.linalg.inv(cov)
correct = 0
#ignore the scalor term in prob.
for i in range(train_x.shape[0]):
    px_c0 = np.exp(-0.5*np.dot(np.dot(train_x[i]-mean_c0,cov_inv),(train_x[i]-mean_c0).T))
    px_c1 = np.exp(-0.5*np.dot(np.dot(train_x[i]-mean_c1,cov_inv),(train_x[i]-mean_c1).T))
    p0 = px_c0*p_c0
    p1 = px_c1*p_c1
    if (p0>p1 and train_y[i]==0) or (p1>p0 and train_y[i]==1):
        correct += 1
train_accuracy = correct/train_x.shape[0]
correct = 0 
for i in range(validation_x.shape[0]):
    px_c0 = np.exp(-0.5*np.dot(np.dot(validation_x[i]-mean_c0,cov_inv),(validation_x[i]-mean_c0).T))
    px_c1 = np.exp(-0.5*np.dot(np.dot(validation_x[i]-mean_c1,cov_inv),(validation_x[i]-mean_c1).T))
    p0 = px_c0*p_c0
    p1 = px_c1*p_c1
    if (p0>p1 and validation_y[i]==0) or (p1>p0 and validation_y[i]==1):
        correct += 1
validation_accuracy = correct/validation_x.shape[0]
print('train_accuray:',train_accuracy)
print('validation_accuracy:',validation_accuracy)

def save_model(mean,std,p_c0,p_c1,mean_c0,mean_c1,cov_inv):
    np.save('model_g.npy',[mean,std,p_c0,p_c1,mean_c0,mean_c1,cov_inv])
save_model(mean,std,p_c0,p_c1,mean_c0,mean_c1,cov_inv)


