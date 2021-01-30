import pandas as pd
import numpy as np
import sys
x_df = pd.read_csv('.\X_test')
x_df = x_df.drop('fnlwgt',axis=1)
test_x = np.array(x_df.values).astype(np.float)
def normalize(test_x,mean,std):
    columns_to_normalized = [0,2,3,4]
    test_x[:,columns_to_normalized] = (test_x[:,columns_to_normalized]-mean)/std
    return test_x
def output_test_result(path,test_y):
    df1 = pd.DataFrame(['label'],index=['id'],columns = None)
    rows = [str(i+1) for i in range(test_y.shape[0])]
    df2 = pd.DataFrame(test_y.astype(np.int),index = rows)
    df = pd.concat([df1,df2])
    df.to_csv(path,header = None) 
    
model = np.load('model.npy')
w = model[0]
b = model[1]
parameter = np.load('parameter.npy')
mean = parameter[0]
std = parameter[1]

test_x = normalize(test_x,mean,std)
test_y = np.dot(test_x,w)+b
test_y[test_y > 0.5] = 1
test_y[test_y <= 0.5] = 0
output_test_result('hi.csv',test_y) 

