import statistics
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import plotly.graph_objects as go
import pandas as pd

###### function to compute the hypothesis values for a given theta vector 

def computeHyp(X,Y,theta):
    A = np.dot(X,theta)    # theta_transpose * X
    A = A*(-1) 
    B = np.exp(A).reshape(-1,1)   # exp(-theta_transpose * X)
    B+=1 
    h = 1/B  # 1/(1+ exp(-theta_transpose * X))
    return h

###### function to compute the log likelihood LL(theta)

def computeLL(X,Y,theta) :
    LL=0 #store the log likelihood
    Y = np.array(Y)
    h = computeHyp(X,Y,theta)
    h = np.array(h)
    h_comp = 1-h
    A = np.log(h)
    B = np.log(h_comp)
    Y_comp = 1 - Y
    LL = np.dot(np.transpose(Y), A) + np.dot(np.transpose(Y_comp), B)
    return LL[0][0]

###### function to compute the gradient of log likelihood LL(theta)

def computeGrad(X,Y,theta):
    h = computeHyp(X,Y,theta)
    del_L = np.dot(np.transpose(X), np.subtract(Y,h))
    return del_L

###### function to compute the Hessian of log likelihood LL(theta)

def computeHessian(X,Y,theta):
    h = computeHyp(X,Y,theta)
    h_comp = 1-h
    m = len(Y)
    D = np.zeros([m,m],dtype = float )
    for i in range(m):
        D[i][i] = h[i] * h_comp[i]
    A = np.dot(np.transpose(X),D)
    H = np.dot(A,X)
    return H    

###### function to learn the parameters using Newton's method

def NewtonMethod(X,Y,theta,alpha) :
    m = len(Y)
    n = len(theta) - 1
    LL_prev = -1
    LL_new = computeLL(X,Y,theta)
    print("Initial likelihood is ", LL_new)
    LL_history = [LL_new]
    thetas_history = [ [] for i in range(n+1)]
    iter_num = 1
    delta = 0
    X_x = np.array(X).reshape(m,-1)   # creating np array to ease slicing
    while (iter_num ==1 or (LL_prev +delta < LL_new )) :
        LL_prev = LL_new      
        del_L = computeGrad(X,Y,theta)
        H = computeHessian(X,Y,theta)
        H_inv = np.linalg.inv(H)
        update = np.dot(H_inv,del_L)
        theta = np.add(theta, update)
        LL_new = computeLL(X,Y,theta)
        LL_history.append(LL_new)
        for i in range(n+1):
            thetas_history[i].append(theta[i][0])
        iter_num+=1
    print("Final likelihood is ", LL_new)

    return (theta, LL_history, thetas_history)


X = []  # to hold the input X values
Y = []  # to hold the input Y values
mean = 0
sigma = 0

######## storing the traning examples  ########

fileX = open("logisticX.csv")
fileY = open("logisticY.csv")
csvReader = csv.reader(fileX)
for row in csvReader:
    x_val = list(map(float, row))
    X.append(x_val)
csvReader = csv.reader(fileY)
for row in csvReader:
    Y.append([int(row[0])])
fileX.close()
fileY.close()

###### Normalizing the data ##########

m = len(X)
n = len(X[0])

mean = np.mean(X,axis = 0) # nx1
sigma = np.std(X,axis = 0) # nx1

X_norm = []
data = []
for i in range(m):
    x_norm = np.subtract(X[i],mean)
    x_norm = np.divide(x_norm, sigma)
    x_val = [1]  # x_0 term
    x_val.extend(x_norm)
    X_norm.append(x_val)
    row_data = x_val + [Y[i][0]]
    data.append(row_data)

data = pd.DataFrame(data)    

###### Using Newton's method to learn theta ######

theta = [[0] for i in range(n+1)] # initialize the fitting parameters
alpha = 0.01 #learning rate

theta, LL_history, thetas_history = NewtonMethod(X_norm,Y,theta,alpha)

print("Theta values are : ")
for i in range(n+1):
    print(theta[i][0])

########## plotting Log likelihood over iterations ########## 

iters = []
for i in range(len(LL_history)):
    iters.append(i)

plt.plot(iters, LL_history)   
plt.xlabel('Iterations')
plt.ylabel('LL (Log likelihood)') 
plt.title('Log likelihood using Newton method')
plt.savefig('LLNewtonMethod')
plt.show()

########## plotting trining data and decision boundary ########## 

print(data)
y = data.iloc[:, -1]

# filter out the examples that had y=1
admitted = data.loc[y == 1]

# filter out the examples that had y=0
not_admitted = data.loc[y == 0]

plt.scatter(admitted.iloc[:, 1], admitted.iloc[:, 2], s=10, label='y = 0 ')
plt.scatter(not_admitted.iloc[:, 1], not_admitted.iloc[:, 2], s=10, label='y = 1')

# decision boundary
intercept = theta[0][0]
slope = theta[1][0]

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, label = "Decision Boundary", color = "red")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Classification using Newton Method')
plt.legend()
plt.savefig('DecisionBoundary_NewtonMethod')
plt.show()

