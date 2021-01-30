import pandas as pd
import numpy as np
import sys
def read_testdata(filename):
    df = pd.read_csv(filename)
    preserved_columns = ['SO2','NOx','NO2','THC','PM10','PM2.5']
    df = df[preserved_columns]
    data = df.values
    data = data.astype(np.float)
    for i in range(data.shape[0]):
        for j in range(len(preserved_columns)):
            if data[i,j] == 0:
                data[i,j] = (data[i-1,j]+data[i+1,j])/2
    print('testdata preprocessed')
    test_x = []
    for i in range(int(data.shape[0]/9)):                
        x = data[9*i:9*i+9,:]
        test_x.append(x.reshape(-1))
    test_x = np.array(test_x)
    return test_x
def load_model():
    model = pd.read_csv('./model#2.csv',header = None)
    w = (model.values[:-1].astype(np.float)).reshape((-1,1))
    b = float(model.values[-1])
    return (w , b)

def output_test_result(path,test_y):
    df1 = pd.DataFrame(['value'],index=['id'],columns = None)
    rows = ['id_'+str(i) for i in range(test_y.shape[0])]
    df2 = pd.DataFrame(test_y,index = rows)
    df = pd.concat([df1,df2])
    df.to_csv(path,header = None)
test_x = read_testdata(sys.argv[1])
w,b = load_model()
test_y = np.dot(test_x,w)+b
output_test_result(sys.argv[2],test_y)    

    
