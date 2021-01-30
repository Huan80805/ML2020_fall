import pandas as pd
import numpy as np
import sys
x_df = pd.read_csv(sys.argv[1])
x_df = x_df.drop('fnlwgt',axis=1)
test_x = np.array(x_df.values).astype(np.float)
def normalize(test_x,mean,std):
    columns_to_normalized = [0,2,3,4]
    test_x[:,columns_to_normalized] = (test_x[:,columns_to_normalized]-mean)/std
    return test_x
def load_model():
    model = np.load('model_g.npy')
    return model[0],model[1],model[2],model[3],model[4],model[5],model[6]

def output_test_result(path,test_y):
    df1 = pd.DataFrame(['label'],index=['id'],columns = None)
    rows = [str(i+1) for i in range(test_y.shape[0])]
    df2 = pd.DataFrame(test_y.astype(np.int),index = rows)
    df = pd.concat([df1,df2])
    df.to_csv(path,header = None)

mean,std,p_c0,p_c1,mean_c0,mean_c1,cov_inv = load_model()
test_x = normalize(test_x,mean,std)
test_y = np.arange(test_x.shape[0]).reshape((-1,1))
for i in range(test_x.shape[0]):
    px_c0 = np.exp(-0.5*np.dot(np.dot(test_x[i]-mean_c0,cov_inv),(test_x[i]-mean_c0).T))
    px_c1 = np.exp(-0.5*np.dot(np.dot(test_x[i]-mean_c1,cov_inv),(test_x[i]-mean_c1).T))
    p0 = px_c0*p_c0
    p1 = px_c1*p_c1
    if p0>p1:
        test_y[i] = 0
    else: 
        test_y[i] = 1
output_test_result(sys.argv[2],test_y) 