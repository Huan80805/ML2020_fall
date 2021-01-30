#logistic regression using sklearn
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
def output_test_result(path,test_y):
    df1 = pd.DataFrame(['label'],index=['id'],columns = None)
    rows = [str(i+1) for i in range(test_y.shape[0])]
    df2 = pd.DataFrame(test_y.astype(np.int),index = rows)
    df = pd.concat([df1,df2])
    df.to_csv(path,header = None)

x_df = pd.read_csv(sys.argv[1])
x_df = x_df.drop('fnlwgt',axis=1)
test_x = np.array(x_df.values).astype(np.float)


scaler = pickle.load(open('scaler_best','rb'))
test_x = scaler.transform(test_x)
model = pickle.load(open('model_best','rb'))
test_y = model.predict(test_x) 
output_test_result(sys.argv[2],test_y)