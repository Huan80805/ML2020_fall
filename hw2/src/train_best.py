##logistic using sklearn
##using StandardScaler or MinMaxScaler?
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
x_df = pd.read_csv('.\X_train')
y_df = pd.read_csv('.\Y_train',header = None)
x_df = x_df.drop('fnlwgt',axis = 1)
train_x = np.array(x_df.values).astype(np.float)
train_y = np.array(y_df.values).astype(np.int).reshape((-1))

dimension = train_x.shape[1]
data_len = train_x.shape[0]//2
validation_x = train_x[data_len:]
validation_y = train_y[data_len:]
train_x = train_x[:data_len]
train_y = train_y[:data_len]

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
validation_x = scaler.transform(validation_x)
#model = GradientBoostingClassifier()
model = GradientBoostingClassifier(learning_rate=0.05,n_estimators=500)
model.fit(train_x,train_y)
print('train accuracy:',model.score(train_x,train_y))
print('validation accuracy:',model.score(validation_x,validation_y))
pickle.dump(model,open('model','wb'))
pickle.dump(scaler,open('scaler','wb'))

