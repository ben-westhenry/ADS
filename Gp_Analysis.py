import time
import pandas as pd
import math as maths
from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import sys
sys.path.insert(0, "/homeb/jb14389")
import GPy

def Gen_Mean(FileAdress):

    data = pd.read_csv(FileAdress, delim_whitespace = True).values

    data = Shift_Data(data)

    x = np.asarray([[a] for a in data[:,0]])

    y = np.asarray([[a] for a in data[:,1]])

    kern = GPy.kern.RBF(1,lengthscale = 2000) + GPy.kern.PeriodicExponential(input_dim=1, variance=1.0, lengthscale=1.0, period=105, n_freq=100, lower=0.5*105, upper=1.5*105)

    m = GPy.models.GPRegression(x, y, kernel = kern) 

    m.optimize()

    print(m)

    m.plot(plot_limits=(0.99*min(x), 1.01*max(x)))

    plt.show()
    
def Shift_Data(data):
    
    mean = [(data[0,0]+data[-1,0])/2,np.mean(data[:,1]),0]
    
    result = [a - mean for a in data]
    
    return np.asarray(result)

Gen_Mean("Raw Data/Ogle/I/LMC562.01.10.dat")
