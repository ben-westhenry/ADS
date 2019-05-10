import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, "/homeb/jb14389") # <-- Change to your GPy location
import GPy

# Subtract the meen flux and shift the y axis so data is centerd about zero
def Shift_Data(data):
    
    mean = [0 for i in range(len(data[0]))]
    
    mean[0] = (data[0,0]+data[-1,0])/2
    mean[1] = np.mean(data[:,1])
    
    result = [a - mean for a in data]
    
    return np.asarray(result)
    
# Use GP regresion to remove any general trends in the data
def Remove_General_Trends(data, plot = False):

    x = np.asarray([[a] for a in data[:,0]])

    y = np.asarray([[a] for a in data[:,1]])

    kern = GPy.kern.RBF(1, lengthscale = 200) 

    m = GPy.models.GPRegression(x, y, kernel = kern) 

    #m.optimize()


    if(plot == True):
        print(m)

        m.plot(plot_limits=(0.99*min(x), 1.01*max(x)))

        plt.show()
    
    GP = m.predict(x)[0]
    
    return np.asarray([[x[i], y[i] - GP[i]] for i in range(len(x))])
    
def Normalise_Amplitude(data, plot = False):

    tmp = cp.deepcopy(data)
    
    for val in tmp:
        val[1] = np.abs(val[1])
    
    kern = GPy.kern.RBF(1, lengthscale = 100) 

    m = GPy.models.GPRegression(tmp[:,0], tmp[:,1], kernel = kern) 

    #m.optimize()


    if(plot == True):
        print(m)

        m.plot(plot_limits=(0.99*min(tmp[:,0]), 1.01*max(tmp[:,0])))

        plt.show()
        
    GP = m.predict(data[:,0])[0]
    
    for i in range(len(data)):
        
        data[i,1] *= GP[0]/GP[i]
        
    return data
    

def Fold(data, P, plot = True):
    
    NewX = []
    
    for point in data:
        NewX.append(point[0]%P)
        
    plt.scatter(NewX,data[:,1],s = 10,c=data[:,0])
    plt.colorbar(sc)
    plt.show()
    
# -- Main --------------------
Plot = True

# Read out data
data = pd.read_csv("Raw Data/Ogle/I/LMC562.01.11.dat", delim_whitespace = True).values # <-- Change to your data location

sc = plt.scatter(data[:,0], data[:,1], s = 10)#, c=data[:,0])
#cbar = plt.colorbar(sc)
plt.show()

# Preproces data;
data = Shift_Data(data)

data = Remove_General_Trends(data, plot = Plot)

data = Normalise_Amplitude(data, plot = Plot)

sc = plt.scatter(data[:,0], data[:,1], s = 10, c=data[:,0])
cbar = plt.colorbar(sc)
plt.show()

Fold(data,195.4)
Fold(data,128.89)
