import statistics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import plotly.graph_objects as go
import pandas as pd

X = []  #to hold the initial X values
Y = []  #to hold the Y values
mean = 0
sigma = 0


######## storing the traning examples  ########

X = np.loadtxt('q4x.dat')    # mx2
ydata = np.loadtxt('q4y.dat', dtype = str)  # mx1
m = len(X)
n = len(X[0])

###### Normalizing the data ##########

mean = np.mean(X,axis = 0) # nx1
sigma = np.std(X,axis = 0) # nx1

X_norm = []  # mxn
for i in range(m):
    x_norm = np.subtract(X[i],mean)
    x_norm = np.divide(x_norm, sigma)
    x_val = [] 
    x_val.extend(x_norm)
    X_norm.append(x_val)

class0_sum = [0,0]  # 1xn
class1_sum = [0,0]  # 1xn
class0_count = 0
class1_count = 0

###### Calculating the parameters #####

Y = []
data = []
for i in range(m):
    if(ydata[i] == 'Alaska'):
        Y.append(0)
        class0_count+=1
        class0_sum[0] += X_norm[i][0]
        class0_sum[1] += X_norm[i][1]
        row_data = X_norm[i] + [Y[i]]
        data.append(row_data)  
    else:
        Y.append(1)
        class1_count+=1
        class1_sum[0] += X_norm[i][0]
        class1_sum[1] += X_norm[i][1]
        row_data = X_norm[i] + [Y[i]]
        data.append(row_data)  

data = pd.DataFrame(data)    
mu_0 = np.array(class0_sum)/class0_count    # 1xn
mu_1 = np.array(class1_sum)/class1_count    # 1xn
mu = [mu_0, mu_1]
phi = class1_count/m

sigma_sum = [[0,0],[0,0]]
for i in range(m):
    v = np.subtract(X_norm[i] , mu[Y[i]])
    update = np.outer(v,v)
    sigma_sum = np.add(sigma_sum,update)
sigma = sigma_sum/m               # nxn

sigma0_sum = [[0,0],[0,0]]
for i in range(m):
    v = np.subtract(X_norm[i] , mu[Y[i]])
    update = (1- Y[i]) * np.outer(v,v)
    sigma0_sum = np.add(sigma0_sum,update)
sigma0 = sigma0_sum/class0_count               # nxn

sigma1_sum = [[0,0],[0,0]]
for i in range(m):
    v = np.subtract(X_norm[i] , mu[Y[i]])
    update = Y[i] * np.outer(v,v)
    sigma1_sum = np.add(sigma1_sum,update)
sigma1 = sigma1_sum/class1_count               # nxn



print( "mu : ")
print(mu)
print("phi = " )
print(phi)
print("sigma :" )
print(sigma)
print("sigma0 : ")
print(sigma0)
print("sigma1 : ")
print(sigma1)

###### Using GDA to calculate the decision boundary  (sigma0 = sigma1 = sigma)  ######

sig_inv = np.linalg.inv(sigma)
A = np.dot(mu_0, sig_inv)
B = np.dot(mu_1, sig_inv)
[a,b] = np.subtract(A,B)
C = np.dot(B,np.transpose(mu_1))
D = np.dot(A,np.transpose(mu_0))
c = C-D + np.log((1-phi)/phi)

###### Using GDA to calculate the decision boundary  (sigma0 != sigma1)  ######

sig0_inv = np.linalg.inv(sigma)
sig1_inv = np.linalg.inv(sigma)
[[p1, p2], [p3,p4]] = sig1_inv - sig0_inv
a0 = p1/2
a1 = p4/4
a2 = (p2 + p3)/2
[ a3, a4] = np.subtract(np.dot(mu_0, sig0_inv), np.dot(mu_1, sig1_inv))

P =np.dot(np.dot(mu_1, sig1_inv),np.transpose(mu_1))
Q =np.dot(np.dot(mu_0, sig0_inv),np.transpose(mu_0))
w = (( np.linalg.det(sigma1) )**0.5)/((np.linalg.det(sigma0))**0.5)
cons = np.log( ((1-phi)/phi) * w)
a5 = P-Q + cons


A1 = np.dot(mu_0, sig_inv)
B1 = np.dot(mu_1, sig_inv)
[a1,b1] = np.subtract(A,B)
C1 = np.dot(B,np.transpose(mu_1))
D1 = np.dot(A,np.transpose(mu_0))
c = C1-D1 + np.log((1-phi)/phi)




########## plotting training data and hypothesis function ########## 

print(data)
y = data.iloc[:, -1]

# filter out the examples corresponding to y = 1 , i.e. Canada
admitted = data.loc[y == 1]

# filter out the examples corresponding to y = 0 , i.e. Alaska
not_admitted = data.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Canada ')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Alaska')

intercept = -c/b
slope = -a/b

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, label = "Linear Decision Boundary", color = "red")

def quadSeparator(phi, mean0, mean1, sigma0, sigma1, x):
    covInv0 = np.linalg.inv(sigma0)
    covInv1 = np.linalg.inv(sigma1)
    sqrtDetCov0 = (np.linalg.det(sigma0))**0.5
    sqrtDetCov1 = (np.linalg.det(sigma1))**0.5
    const = np.log(((1-phi)/phi)*(sqrtDetCov1/sqrtDetCov0)) + ((mean1.T).dot(covInv1).dot(mean1) - (mean0.T).dot(covInv0).dot(mean0))/2
    nonConst = (x.T).dot(covInv1-covInv0).dot(x)/2 - ((mean1.T).dot(covInv1) - (mean0.T).dot(covInv0)).dot(x)
    sol = nonConst + const
    return sol

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x,y, indexing = 'xy')
z = np.zeros((x.size, y.size))
for (i,j), v in np.ndenumerate(z):
    z[i,j] = quadSeparator(phi, mu_0, mu_1, sigma0, sigma1, np.array([[X[i,j]],[Y[i,j]]]))

axes.contour(X,Y,z,levels = [0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear and Quadratic Separator using GDA')
plt.legend()
plt.show()




