import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

data = pd.read_csv("train.csv")
data = data.dropna()
x= np.array(data["x"])
o=x
y=np.array(data["y"])




x=x.reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)

theta=np.array([1.0,1.0])
alpha=0.001

def cost(x,y,theta):
    yhat=np.array(x@theta)
    inner=np.power((yhat-y),2)
    error= np.nansum(inner)/(2*len(x))
    return error


def gradient_descent(x,y,theta):
    iter=1000
    for i in range(iter):
        theta=theta-(alpha/len(x))*(np.nansum((x@theta-y)*x,axis=0))
        error2=cost(x,y,theta)
        if i%100==0:
            print(error2," ",theta)
    
    
gradient_descent(x,y,theta)
