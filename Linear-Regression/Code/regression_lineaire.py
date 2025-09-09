import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/adrie/Desktop/IA/AI-Journey/Linear-Regression/Dataset/housing.csv')
print(data)

y = data["median_house_value"]
x = data["median_income"]

n = len(x)

x_mean = np.sum(x)/n
y_mean = np.sum(y)/n

a = (np.sum((x-x_mean)*(y-y_mean)))/np.sum((x-x_mean)**2)
print(a)
b = y_mean-a*x_mean
print(b)

plt.scatter(x,y, alpha = 0.5)
plt.plot(x,a*x+b, color="red")
"""plt.xlabel("Ocean Proximity")
plt.ylabel("Median Income")
plt.legend()"""
plt.show()