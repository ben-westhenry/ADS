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
    
# Calculate fourier transform
def Calc_Xk(xn,pn,wk):

    return [np.sum(xn*np.e**(-2*np.pi*1j*pn*wk[i]))/len(wk) for i in range(len(wk))]

# Calculate inverse fourier transform
def Calc_xn(Xk,pn,wk):
    
    return [np.sum(Xk*np.e**( 2*np.pi*1j*pn[i]*wk))         for i in range(len(pn))]
    
def Calc_Period(data, P_range_min, P_range_max, sample_points, plot = False):

    MaxF = 0.1

    wk = np.linspace(P_range_max**(-1), P_range_min**(-1),num = sample_points)

    Xk = Calc_Xk(data[:,1],data[:,0],wk) # Calculate fourier transform
    
    if(plot == True):
        plt.plot(wk,Xk)
        plt.show()

    if(max(Xk) > -min(Xk)):
        return wk[Xk.index(max(Xk))]**-1

    else:
        return wk[Xk.index(min(Xk))]**-1


def Est_Period_Error(data, P, P_range_min, P_range_max, sample_points, Iterations, Print = False):

    P_Error = 0
    
    
    for i in range(Iterations):
        Sub_Sampled_data = []
        
        for j in np.random.randint(0, high = len(data), size = len(data)):
            Sub_Sampled_data.append(data[j])
            
        new_P = Calc_Period(np.asarray(Sub_Sampled_data), P_range_min, P_range_max, sample_points)
        
        if(Print == True):
            print(100 * i/Iterations, "% P = ", new_P)
        
        P_Error = P_Error + (P - new_P)**2
        
    return (P_Error/Iterations)**0.5
    

# -- Main -------------------
Plot = True

# Read out data
data = pd.read_csv("Sinx/Sinx011.csv", delim_whitespace = False).values # <-- Change to your data location

# Preproces data;
data = Shift_Data(data)

data = Remove_General_Trends(data, plot = Plot)

# Plot new form of data;
if(Plot == True):
    plt.scatter(data[:,0],data[:,1],s = 10)

    plt.show()

P = Calc_Period(data, 10, 500, 10000, plot = Plot)

print(P)

P_Error = Est_Period_Error(data, P, 0.9*P, 1.1*P, 1000, 100, Print = Plot)

print(P_Error)
