import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

data = pd.read_csv("train.csv")
data = data.dropna()
x= np.array(data["x"])

y=np.array(data["y"])
+
x=x.reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)
o=np.array(x[:,1])

theta=np.array([1.0,1.0])
alpha=0.00001

def cost(x,y,theta):
    yhat=np.array(x@theta)
    inner=np.power((yhat-y),2)
    loss= np.nansum(inner)/(2*len(x))
    return loss


def gradient_descent(x,y,theta):
    iter=10000
    for i in range(iter):
        theta=theta-(alpha/len(x))*(np.nansum((x@theta-y)*o,axis=0))
        loss2=cost(x,y,theta)
        if i%100==0:
            print(loss2," ",theta)
            
        
    
    
gradient_descent(x,y,theta)
