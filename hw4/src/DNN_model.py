#set up DNN_model

import torch.nn as nn
class DNN(nn.Module):
    def __init__(self , dimension):
        super(DNN , self).__init__()
        self.linear = nn.Sequential(
			nn.Linear(dimension , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() 
		)
        self.classifier = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(100, 1),
            nn.Sigmoid() 
        )

    def forward(self , x):
        x = self.linear(x)
        x = self.classifier(x)
        return x
