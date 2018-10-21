# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 01:17:19 2018

@author: CAPTAIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)


data,x,y,x1,y1,alpha,theta,one,one1=None,None,None,None,None,None,None,None,None



def data_preprocess():
    global x,y,x1,y1,data,alpha,theta,one,one1
    data= pd.read_csv("train.csv")
    data=data.dropna()
    x=np.array(data["x"])
    y=np.array(data["y"])
    theta=np.random.randn(2,1)*0.01
    alpha=0.0001  
    data1=pd.read_csv("test.csv")
    data1=data1.dropna()
    x1=np.array(data1["x"])
    y1=np.array(data1["y"])
    one1=np.ones((x1.shape[0],1))
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)
    x1=x1.reshape(-1,1)
    y1=y1.reshape(-1,1)
    one=np.ones((x.shape[0],1))
    x=np.hstack((one,x))
    x1=np.hstack((one1,x1))


def grad_descent(x,y,alpha):
    
        t=x.T
        for i in range(10000):
            global theta,yhat,c2
            yhat=np.matmul(x,theta)
            theta=theta-(alpha/x.shape[0])*(np.matmul(t,(yhat-y)))
            c2=cost(x,y,alpha)
            if(i%100==0):
                print(c2)
        return theta
            
            
def cost(x,y,alpha):
    yhat=np.matmul(x,theta)
    sub=np.subtract(yhat,y)
    c1= (1/(2*x.shape[0]))*(np.sum(np.power(sub,2)))
    return c1

data_preprocess()
b=grad_descent(x,y,alpha)

ui=np.matmul(x1,b)

plt.scatter(x1[:,1], y1)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0][0] + theta[1][0]* x_vals #the line equation
plt.plot(x_vals,y_vals, '--')
    
    
