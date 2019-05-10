import numpy as np
import scipy.signal as sps
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
    
def Calc_Dispersion(data, P, Bin_Count, plot = False):

    Bins = [[] for i in range(Bin_Count)]

    Result = 0.
    
    if(plot == True):
        plt.scatter(data[:,0]%P, data[:,1], s = 1)
        plt.show()


    for point in data:
    
        bin_num = int(((400+point[0])%P)*Bin_Count/P)
        
        Bins[bin_num].append(point[1,0])
        
    for i in range(Bin_Count):
        if(len(Bins[i]) > 1):
            Result += (len(Bins[i])-1)*np.std(Bins[i])**2
        
    return Result
    
def Find_Period_From_Points(points, minp, maxp, sample_count = 100, plot = False):

    print(points)

    if(len(points) == 1):
        return points[0]
        
    if(len(points) == 0):
        print('Error; Find_Period_From_Points; no points given returning 1')
        return 1

    stds = []

    for p in np.linspace(minp,maxp,sample_count):
    
        std = 0
        
        
        
        for val in points:
            std += (val-p*int(0.5+val/p))**2
        
        stds.append((std**0.5-p)/p)
        
    if(plot == True):
        plt.scatter(np.linspace(minp,maxp,100),stds)
        
        plt.show()
    
    return (np.linspace(minp,maxp,sample_count)[[i for i in range(len(stds)) if stds[i] == min(stds)]])[0]
    
def Est_Period_Error(points, P, P_range_min, P_range_max, sample_count, Iterations, Print = False):

    P_Error = 0
    
    
    for i in range(Iterations):
        Sub_Sampled_data = []
        
        for j in np.random.randint(0, high = len(points), size = len(points)):
            Sub_Sampled_data.append(points[j])
            
            
        new_P = Find_Period_From_Points(Sub_Sampled_data, P_range_min, P_range_max, sample_count=sample_count)
        
        if(Print == True):
            print(100 * i/Iterations, "% P = ", new_P)
        
        P_Error = P_Error + (P - new_P)**2
        
    return (P_Error/Iterations)**0.5
    
def Calc_Period(data, P_range_min, P_range_max, sample_points, plot = False):

    P = 1
    
    Dispersion = np.asarray([[0.1,0.1] for i in range(sample_points)])
    
    for i in range(sample_points):
    
        Dispersion[i,0] = P_range_min + (P_range_max-P_range_min)*i/sample_points
    
        Dispersion[i,1] = Calc_Dispersion(data, Dispersion[i,0], 100)
        
    # This algorithm finds peeks so we need to apply a minus to get troughs
    peeks = sps.find_peaks_cwt(-Dispersion[:,1],widths = [5,10,20,30,40,50])
    
    if(plot == True):
        plt.scatter(Dispersion[  :  ,0],Dispersion[  :  ,1], s =  1)
        plt.scatter(Dispersion[tuple(peeks),0],Dispersion[tuple(peeks),1], s = 10)
        
    
    tmp = []
    
    for val in peeks:
        print(-Dispersion[val,1] , 2*max(-Dispersion[tuple(peeks),1]))
        if(-Dispersion[val,1] > 2*max(-Dispersion[tuple(peeks),1])):
            tmp.append(val)
    
    peeks = tmp
    
    
    if(plot == True):
        #plt.scatter(Dispersion[  :  ,0],Dispersion[  :  ,1], s =  1)
        plt.scatter(Dispersion[tuple(peeks),0],Dispersion[tuple(peeks),1], s = 10)
        plt.show()
    
    peeks = Dispersion[tuple(peeks),0]
    
    P = Find_Period_From_Points(peeks,0.1*min(peeks),2*min(peeks),plot = plot)
    
    P_Error = Est_Period_Error(peeks, P, 0.8*P, 1.2*P, 100, 100,Print = plot)
    
    return P, P_Error



    
def Fold(data, P, plot = True):
    
    plt.scatter(data[:,0],data[:,1],s = 10,c=data[:,0])
    plt.show()
    
    NewX = []
    
    for point in data:
        NewX.append(point[0]%P)
        
    plt.scatter(NewX,data[:,1],s = 10,c=data[:,0])
    plt.show()

# -- Main -------------------
Plot = True

# Read out data
data = pd.read_csv("Sin_sqrx/Sin_sqrx005.csv", delim_whitespace = False).values # <-- Change to your data location

# Plot new form of data;
if(Plot == True):
    plt.scatter(data[:,0],data[:,1],s = 10)

    plt.show()

# Preproces data;
data = Shift_Data(data)

# Plot new form of data;
if(Plot == True):
    plt.scatter(data[:,0],data[:,1],s = 10)

    plt.show()

data = Remove_General_Trends(data, plot = Plot)

# Plot new form of data;
if(Plot == True):
    plt.scatter(data[:,0],data[:,1],s = 10)

    plt.show()

data = Normalise_Amplitude(data, plot = Plot)

# Plot new form of data;
if(Plot == True):
    plt.scatter(data[:,0],data[:,1],s = 10)

    plt.show()

P, P_Error = Calc_Period(data, 0.01, 500, 1000, plot = Plot)

print(P, P_Error)

Fold(data,P)
