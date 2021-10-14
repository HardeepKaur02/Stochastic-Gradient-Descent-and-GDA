import statistics
import csv
import numpy as np
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

###### helper function (to assist plotting) to compute the cost matrix 'costs' for meshgrid X1 and X2 containing theta_0 and theta_1 values 

def cost(X,Y,X1,X2):
    n = len(X1)
    costs = [ [0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            theta = [[X1[i][j]],[X2[i][j]]]
            costs[i][j] = computeCost(X,Y,theta)
    costs = np.array(costs).reshape(n,n)        
    return costs        


###### function that performs gradient descent to learn theta

def gradientDescent(X,Y,theta,alpha,delta) :
    m = len(Y)
    n = len(theta) - 1
    J_prev = -1
    J_new = computeCost(X,Y,theta)
    print("Initial cost is ", J_new)
    J_history = [J_new]
    thetas_history = [ [0] for i in range(n+1)]
    iter_num = 1


    while (iter_num ==1 or abs(J_new - J_prev) > delta) :
        J_prev = J_new        
        A = np.dot(X, theta)
        B = np.subtract(A, Y)

        C = np.dot(np.transpose(X),B)
        C*= alpha
        C/= m           
        theta = np.subtract(theta,C)
        J_new = computeCost(X,Y,theta)
        J_history.append(J_new)
        for i in range(n+1):
            thetas_history[i].append(theta[i][0])
        iter_num+=1
    print("Final cost is ", J_new)

    return (theta, J_history, thetas_history)


X = []  #to hold the initial X values
Y = []  #to hold the Y values
mean = 0
sigma = 0


######## storing the traning examples and calculating the mean of X values ########

fileX = open("linearX.csv")
fileY = open("linearY.csv")
csvReader = csv.reader(fileX)
for row in csvReader:
    x_val = float(row[0])
    X.append(x_val)
csvReader = csv.reader(fileY)
for row in csvReader:
    Y.append([float(row[0])])
fileX.close()
fileY.close()

###### Normalizing the data ##########

m = len(X)

mean = statistics.mean(X)
sigma = statistics.stdev(X, xbar = mean)

X_norm = []
X1_norm =[]
for i in range(m):
    x_val_norm = (X[i]-mean)/sigma
    X_norm.append([1,x_val_norm])
    X1_norm.append([x_val_norm])

###### Using linear regression and gradient descent to learn theta ######

theta = [[0],[0]] # initialize the fitting parameters
n = len(theta)

alpha = 0.1 #learning rate
delta = 0.000000000001 #to decide convergence
theta, J_history, thetas_history = gradientDescent(X_norm,Y,theta,alpha,delta)

print("Theta values are : ")
for i in range(n):
    print( "Theta_" + str(i) + " = "+ str(theta[i][0]))


theta_0 = thetas_history[0]
theta_1 = thetas_history[1]

########## plotting training data and hypothesis function ########## 
########## Question 1, part b #########

'''
intercept = theta[0][0]
slope = theta[1][0]
plt.scatter(X1_norm,Y,label = "Normalized Training Data", color = "blue", marker = "*", s = 30)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, label = "Hypothesis", color = "red")
plt.xlabel('Acidity of Wine')
plt.ylabel('Density of Wine')
plt.title('Linear Regression with Gradient Descent. Learning Rate = ' + '{:.0e}'.format(delta) )
plt.legend()
plt.savefig('LinearRegHypothesis')
plt.show()
'''

######### plotting 3d mesh for cost function along with animation of gradent descent############
########## Question 1, part b #########
'''
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

theta_0 = np.linspace(start = 0, stop = 2, num = 100)
theta_1 = np.linspace(start = -1, stop = 1, num = 100)
theta_0,theta_1 = np.meshgrid(theta_0,theta_1)

fig = plt.figure()
ax = fig.gca(projection = '3d')
plt.title('Variation of error value on iterations of Gradient Descent. Learning Rate = ' + '{:.0e}'.format(delta) )
ax.set_xlabel('theta_0 (Intercept)')
ax.set_ylabel('theta_1 (slope)')
ax.set_zlabel('Error ')
ax.set_xlim3d([0.0, 2.0])
ax.set_ylim3d([-1.0, 1.0])
ax.set_zlim3d([0.0, 1.0])
ax.plot_surface(theta_0, theta_1,cost(X_norm,Y,theta_0, theta_1),alpha = 0.6)


N = len(J_history)
lineData = np.empty((3,N))
for i in range(N):
    lineData[0][i] = thetas_history[0][i]
    lineData[1][i] = thetas_history[1][i]
    lineData[2][i] = J_history[i]
data = [lineData]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

line_ani = animation.FuncAnimation(fig, update_lines,N , fargs=(data, lines),interval=50, blit=False)

writervideo = animation.FFMpegWriter(fps=60)
line_ani.save('ErrorAnimation.mp4', writer=writervideo)
plt.show()
'''

######### plotting animation for error function on contours ############
########## Question 1, part c #########

'''
N = len(J_history)

T1 = np.linspace(start = 0, stop = 2, num = 100)
T2 = np.linspace(start = -1, stop = 1, num = 100)
T1,T2 = np.meshgrid(T1,T2)

Z = cost(X_norm,Y,T1,T2)
Z = np.array(Z).reshape(T1.shape)

fig= plt.figure()
ax = fig.gca()
ax.contour(T1, T2, Z, 100, cmap = 'jet')

ax.set_xlabel('theta_0 (Intercept)')
ax.set_ylabel('theta_1 (slope)')
ax.set_xlim([0.0, 2.0])
ax.set_ylim([-1.0, 1.0])


line, = ax.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point, = ax.plot([], [], '*', color = 'red', markersize = 4)
value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')

    return line, point, value_display

def animate_1(i):
    line.set_data(theta_0[:i], theta_1[:i])
    point.set_data(theta_0[i], theta_1[i])
    value_display.set_text('Cost = ' + str(J_history[i]))
    return line, point, value_display

ax.legend(loc = 1)
anim1 = animation.FuncAnimation(fig, animate_1, init_func=init_1,frames=N, interval=100, repeat_delay=60, blit=True)
writervideo = animation.FFMpegWriter(fps=60)
string = 'ContourAnimation_' + str(alpha) + '.mp4'
anim1.save(string, writer=writervideo)

plt.show()

'''