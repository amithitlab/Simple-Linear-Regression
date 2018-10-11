import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)              //to suppress scientific format

data = pd.read_csv("train.csv")                 //reading the dataset
data = data.dropna()                            //to remove NAN values from the dataset
x= np.array(data["x"])

y=np.array(data["y"])
x=x.reshape(-1,1)                               //reshaping the array to vector
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)
o=np.array(x[:,1])                              //concatenating x into one

theta=np.array([1.0,1.0])                        //error matrix
alpha=0.00001                                   //learning rate

def cost(x,y,theta):                            //loss function
    yhat=np.array(x@theta)
    inner=np.power((yhat-y),2)
    loss= np.nansum(inner)/(2*len(x))
    return loss


def gradient_descent(x,y,theta):                //gradient_descent to minimise loss
    iter=10000
    for i in range(iter):
        theta=theta-(alpha/len(x))*(np.nansum((x@theta-y)*o,axis=0))
        loss2=cost(x,y,theta)
        if i%100==0:
            print(loss2," ",theta)              //printing loss
            
        
    
    
gradient_descent(x,y,theta)
