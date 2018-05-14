# Importing Necessary Libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#set output figures dimentions
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Reading Data
data = pd.read_csv('headbrain.csv')
#show the data's shape
print(data.shape)
#set the head of the data
data.head()

# X is the head size
X = data['Head Size(cm^3)'].values
#Y is the brain weight
Y = data['Brain Weight(grams)'].values

# Mean of X
mean_x = np.mean(X)
#mean of Y
mean_y = np.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate b1 and b2
numer = 0
denom = 0
#for each head size measurement
for i in range(m):
    #scale factor numerator
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    #scale factor denomenator
    denom += (X[i] - mean_x) ** 2
#scale factor
b1 = numer / denom
#bias
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print(b1, b0)

# Plotting Values and Regression Line
#allow for a large max
max_x = np.max(X) + 100
#allow for a small minimum
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
#find y using the coeficients
y = b0 + b1 * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

#label x as head size
plt.xlabel('Head Size in cm3')
#label y as brain weight
plt.ylabel('Brain Weight in grams')
#show legend in plot
plt.legend()
#show plot out
plt.show()

# Calculating Root Mean Squares Error
rmse = 0
#loop throug each x,y pair
for i in range(m):
    #find the prediction according to the line
    y_pred = b0 + b1 * X[i]
    #add the squared distance from the line to running sum
    rmse += (Y[i] - y_pred) ** 2
#calculate total root mean squared error
rmse = np.sqrt(rmse/m)
#output the root mean squared error
print(rmse)

# total sum of squares
ss_t = 0
#sum of squares residuals
ss_r = 0
#loop throug each pair of data points
for i in range(m):
    #get prediction based on line
    y_pred = b0 + b1 * X[i]
    #add to running ss total
    ss_t += (Y[i] - mean_y) ** 2
    #add to running resisduals total
    ss_r += (Y[i] - y_pred) ** 2
#calculate rs using the total and the residuals
r2 = 1 - (ss_r/ss_t)
#output r2
print(r2)
