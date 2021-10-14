import statistics
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import plotly.graph_objects as go
import matplotlib.animation as animation

###### function to compute the cost J(theta)

def computeCost(X,Y,theta) :
    m = len(Y) #no. of training examples
    J=0 #store the cost
    A = np.dot(X,theta)
    B = np.subtract(A,Y)
    c = np.multiply(B,B)
    J = np.sum(c)
    J/= (2*m)
    return J

###### function that performs gradient descent to learn theta

def stochasticGradientDescent(X_in,Y_in,theta,alpha,r,k,delta) :
    
    m = len(X_in)
    n = len(X_in[0]) - 1
    num_batches = m//r

    X = np.array(X_in).reshape(m,-1)   # creating np arrays to ease slicing    
    Y = np.array(Y_in).reshape(m,1)

    J_prev = -1
    J_new = computeCost(X,Y,theta)
    print("Initial cost is ", J_new)
    J_history = []
    J_new=-1
    batch_num = 0
    num_iters = 0
    thetas_history = [[0] for i in range(n+1) ]
    
    while(J_prev==-1 or abs(J_new - J_prev) > delta):
        batch_start = batch_num*r
        batch_end = batch_start + r
        J_prev = J_new
        iter_num =0 
        cost_sum = 0
        while(iter_num < k):  #1000, 5000
            X_batch = (X[batch_start:batch_end, :]).reshape(r,-1)
            Y_batch = (Y[batch_start:batch_end, :]).reshape(r,1)
            J_batch = computeCost(X,Y,theta)
            cost_sum+= J_batch
            num_iters+=1

            A = np.dot(X_batch, theta)
            B = np.subtract(A, Y_batch)
            C = np.dot(np.transpose(X_batch),B)
            C*= alpha
            C/= r
            theta = np.subtract(theta,C)
            for i in range(n+1):
                thetas_history[i].append(theta[i])

            batch_num = (batch_num+1)%num_batches
            iter_num+=1
        J_new = cost_sum/k
        J_history.append(J_new)
    print("Final cost is ", J_new)

    return (theta, J_history,num_iters,thetas_history)



######## storing the traning examples  ########

X = []  # to hold the input X values
Y = []  # to hold the input Y values

fileX = open("Sample_X.csv")
fileY = open("Sample_Y.csv")
csvReader = csv.reader(fileX)
for row in csvReader:
    x_val = [1] + list(map(float, row))
    X.append(x_val)
csvReader = csv.reader(fileY)
for row in csvReader:
    Y.append([float(row[0])])
fileX.close()
fileY.close()

m = len(X)
n = len(X[0])-1

#dim(X) = [mx(n+1)]
#dim(Y) = [mx1]

###### randomly shuffle data #####

temp = list(zip(X,Y))
random.shuffle(temp)
X,Y = zip(*temp)

###### Using linear regression and stochastic gradient descent to learn theta ######

theta = [[0] for i in range (n+1)] # initialize the fitting parameters
alpha = 0.001 #learning rate
batch_size = 1 #r
k = 1000
delta = 0.1
num_batches = m//batch_size

theta, J_history,num_iters,thetas_history = stochasticGradientDescent(X,Y,theta,alpha,batch_size,k,delta)
print('Batch size, r = %i' %batch_size)
print("k = %i" %k)
print("delta = %1.6f" %delta)
for i in range(n+1):
    print("theta" + str(i)+ " = " + str(theta[i][0]) )
print("#iterations = %i" %num_iters)

########## testing on test data #########
X_test = []  # to hold the input X values
Y_test = []  # to hold the input Y values

file_test = open("q2test.csv")
csvReader = csv.reader(file_test)
first_row = 1
for row in csvReader:
    if first_row == 0:
        test_row =  list(map(float, row))
        x_val = [1] + test_row[:2]
        y_val = [test_row[-1]]
        X_test.append(x_val)
        Y_test.append(y_val)
    first_row = 0    
file_test.close()

theta = [[3],[1],[2]]
err = computeCost(X_test,Y_test,theta) 
print("Error on test data = " + str(err))


iters = []
for i in range(len(J_history)):
    iters.append(i)

########## plotting cost history averaged over k iterations  ########## 

string  = 'Batch_Size_' + str(batch_size)  + '_k_' + str(k)    
plt.plot(iters, J_history)   
plt.xlabel('Iterations')
plt.ylabel('J (Cost)') 
plt.title('Plot for batch size = %i, ' %batch_size + 'k = %i, ' %k + 'delta = %1.6f' %delta)
plt.show()
plt.savefig(string)

######### plotting 3d mesh for cost function ############


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

theta_0 = np.linspace(start = 2, stop = 4, num = 100)
theta_1 = np.linspace(start = 0, stop = 2, num = 100)
theta_0,theta_1 = np.meshgrid(theta_0,theta_1)

fig = plt.figure()
ax = fig.gca(projection = '3d')
plt.title('Movement of theta values. Batch Size = %i' %batch_size )
ax.set_xlabel('theta_0 ')
ax.set_ylabel('theta_1 ')
ax.set_zlabel('theta_2 ')
ax.set_xlim3d([-2.0, 4.0])
ax.set_ylim3d([0.0, 2.0])
ax.set_zlim3d([1.0, 3.0])
#ax.plot_surface(theta_0, theta_1,cost(X_norm,Y,theta_0, theta_1),alpha = 0.6)


N = num_iters
lineData = np.empty((3,N))
for i in range(N):
    lineData[0][i] = thetas_history[0][i]
    lineData[1][i] = thetas_history[1][i]
    lineData[2][i] = thetas_history[1][i]
data = [lineData]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

line_ani = animation.FuncAnimation(fig, update_lines,N , fargs=(data, lines),interval=50, blit=False)

writervideo = animation.FFMpegWriter(fps=60)
string = 'ThetaAnimation_BatchSize_'+str(batch_size) + '.mp4'
line_ani.save(string, writer=writervideo)
plt.show()
