from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('xclara.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#have arrays of the SSE for each 
sse2 = np.zeros(0)
sse3 = np.zeros(0)
sse4 = np.zeros(0)
sse5 = np.zeros(0)
sse6 = np.zeros(0)
sse7 = np.zeros(0)
    
    #helper function for adding in the sse to the appropiate array
def insertSSE(x,er):
    if x == 2:
        global sse2
        sse2 = np.append(sse2, er)
    elif x == 3:
        global sse3
        sse3 = np.append(sse3, er)
    elif x == 4:
        global sse4
        sse4 = np.append(sse4, er)
    elif x == 5:
        global sse5
        sse5 = np.append(sse5, er)
    elif x == 6:
        global sse6
        sse6 = np.append(sse6, er)
    else:
        global sse7
        sse7 = np.append(sse7, er)

#helper for condesing the SSE into a float
def getSSE(ar):
    #calculate the mean
    mean = np.sum(ar) / ar.size
    # create another array of equal size to keep squared errors
    diff = np.zeros(ar.size)
    for i in range(ar.size):
        diff[i] = (mean - ar[i]) * (mean - ar[i])
    #return the sum of squared errors
    return np.sum(diff)

def run(k):
    # X coordinates of random centroids
    C_x = np.random.randint(0, np.max(X)-20, size=k)
    # Y coordinates of random centroids
    C_y = np.random.randint(0, np.max(X)-20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    print("Initial Centroids")
    print(C)

    # Plotting along with the Centroids
    plt.scatter(f1, f2, c='#050505', s=7)
    plt.scatter(C_x, C_y, marker='*', s=200, c='g')

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)


    #run algorithm loop
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
        #append to error array for later calculation
        insertSSE(k,error)
    
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    
run(7)
print(getSSE(sse7))
