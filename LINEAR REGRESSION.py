import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

data = pd.read_csv("train.csv")

x= np.array(data["x"])
y=np.array(data["y"])

x=x.reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)

theta=np.array([1.0,1.0])



def cost(x,y):
    inner=np.power(((x@theta)-y),2)
    return np.nansum(inner)/(2*len(x))
    
l=cost(x,y)
print(l)