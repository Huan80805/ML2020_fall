import pandas as pd
import numpy as np
import sys
x_df = pd.read_csv(sys.argv[1])
x_df = x_df.drop('fnlwgt',axis=1)
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
test_x = np.array(x_df.values).astype(np.float)


def output_test_result(path,test_y):
    df1 = pd.DataFrame(['label'],index=['id'],columns = None)
    rows = [str(i+1) for i in range(test_y.shape[0])]
    df2 = pd.DataFrame(test_y.astype(np.int),index = rows)
    df = pd.concat([df1,df2])
    df.to_csv(path,header = None)

model = np.load('model.npy')
w = model[0]
b = model[1]

test_y = np.dot(test_x,w)+b
test_y[test_y > 0.5] = 1
test_y[test_y <= 0.5] = 0
output_test_result(sys.argv[2],test_y) 

    
