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
    
    mean = [(data[0,0]+data[-1,0])/2,np.mean(data[:,1]),0]
    
    result = [a - mean for a in data]
    
    return np.asarray(result)
    
# Use GP regresion to remove any general trends in the data
def Remove_General_Trends(data):

    x = np.asarray([[a] for a in data[:,0]])

    y = np.asarray([[a] for a in data[:,1]])

    kern = GPy.kern.RBF(1,lengthscale = 2000) 

    m = GPy.models.GPRegression(x, y, kernel = kern) 

    m.optimize()

    print(m)

    m.plot(plot_limits=(0.99*min(x), 1.01*max(x)))

    plt.show()
    
    GP = m.predict(x)[0]
    
    return np.asarray([[x[i], y[i] - GP[i]] for i in range(len(x))])


# Calculate fourier transform
def Xk(xn,pn,wk):

    return [np.sum(xn*np.e**(-2*np.pi*1j*pn*wk[i]))/len(wk) for i in range(len(wk))]

# Calculate inverse fourier transform
def xn(Xk,pn,wk):
    
    return [np.sum(Xk*np.e**( 2*np.pi*1j*pn[i]*wk))         for i in range(len(pn))]
    
# Read out data
data = pd.read_csv("Raw Data/Ogle/I/LMC562.01.10.dat", delim_whitespace = True).values # <-- Change to your data location

# Preproces data;
data = Shift_Data(data)

data = Remove_General_Trends(data)

# Plot new form of data;
plt.scatter(data[:,0],data[:,1],s = 10)

plt.show()

# Fourier transform data;
count = 10000

MaxF = 0.1

wk = [MaxF*(i/count) for i in range(count)] # Setup test frequencies

Xk = Xk(data[:,1],data[:,0],wk) # Calculate fourier transform

# Plot results
plt.plot(wk, np.abs(Xk))

plt.show()


