import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

data = pd.read_csv("train.csv")
data = data.dropna()
x= np.array(data["x"])


y=np.array(data["y"]).reshape(699,1)
x=x.reshape(-1,1)
one=np.ones((x.shape[0],1))
x=np.concatenate((one,x),axis=1)
o=np.array(x[:,1])


theta=np.random.randn(2,1)*0.001
print(theta.shape)
alpha=0.0001


def cost(x,y,theta):
    yhat=np.array(x@theta)

    inner=np.power((yhat-y),2)
    loss= np.nansum(inner)/(2*len(x))
    return loss


def gradient_descent(x,y,alpha):
    global theta
    iter=10000
    t=x.T
    m=x.shape[0]
    yhat=np.matmul(x,theta)
    theta=theta-(alpha/m)*(np.matmul(t,(yhat-y)))
    return cost(x,y,theta)
cost2=[]   
for i in range(10000):
    cost1=cost(x,y,theta)
    
    gradient_descent(x,y,alpha)
    if i%100==0:
        print (cost(x,y,theta))
        cost2.append(cost1)
   
            
a=gradient_descent(x,y,0.001)
plt.plot(cost2)
plt.xlabel("Experience")
plt.ylabel("Salary in K")


print(np.matmul(x[:5,:],theta),y[0:5])
