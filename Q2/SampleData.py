import numpy as np
import csv 

x1_mu = 3
x1_sigma = 4

x2_mu = -1
x2_sigma = 4

noise_mu = 0
noise_sigma = 2

sample_size = 1000000

theta = np.array([3,1,2]).reshape(-1,1)

X1_sample = np.random.normal(x1_mu,x1_sigma,sample_size)
X2_sample = np.random.normal(x2_mu,x2_sigma,sample_size)
noise_sample = np.random.normal(noise_mu,noise_sigma,sample_size)

fileX = open("Sample_X.csv", 'w')
fileY = open("Sample_Y.csv", 'w')

writer_X = csv.writer(fileX)
writer_Y = csv.writer(fileY)

for i in range(sample_size):
    X_i = [1, X1_sample[i], X2_sample[i]]
    y_i = np.dot(X_i,theta) + noise_sample[i]
    X_i = X_i[1:]
    writer_X.writerow(X_i)
    writer_Y.writerow(y_i)
fileX.close()
fileY.close()

print(abs(x1_mu - np.mean(X1_sample)))
print(abs(x2_mu - np.mean(X2_sample)))