import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
from os import listdir
from os.path import isfile, join
import sys
from astropy.stats import LombScargle
import seaborn; seaborn.set()
from gatspy.periodic import LombScargleFast
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

def Calc_Period(time, flux, error):
    
    #    fig, ax = plt.subplots()
    #   ax.errorbar(time, flux, error, fmt='.k', ecolor='gray')
    #   ax.set(xlabel='Time (days)', ylabel='magitude',
    #   title='LINEAR object {0}')
    # ax.invert_yaxis();

    model = LombScargleFast().fit(time, flux, error)
    periods, power = model.periodogram_auto(nyquist_factor=100)

    #fig, ax = plt.subplots()
    #   ax.plot(periods, power)
#   ax.set(xlim=(2, 400), ylim=(0, 0.8),
#   xlabel='period (days)',
#    ylabel='Lomb-Scargle Power');
#    plt.show()

    # set range and find period
    model.optimizer.period_range=(10, 500)
    period = model.best_period
    # print("period = {0}".format(period))
    #print("error = {0}".format(error))
    return period

def Est_Period_Error(data, data_error, P, P_range_min, P_range_max, sample_points, Iterations, Print = False):

    P_Error = 0
    
    
    for i in range(Iterations):
        print("Error iter; ", i)
    
        Sub_Sampled_data = []
        
        for j in np.random.randint(0, high = len(data), size = len(data)):
            Sub_Sampled_data.append(data[j])
        
        
        
        new_P = Calc_Period(data[:,0],data[:,1],data_error)
        
        if(Print == True):
            print(100 * i/Iterations, "% P = ", new_P)
        
        P_Error = P_Error + (P - new_P)**2
        
    return (P_Error/Iterations)**0.5
    
def Calc_Period_For_All(Files_Path, Output_File_Address, P_range_min, P_range_max, sample_points, Error_iters, Error_sample_points, Plot = False, delim_whitespace = True):
    
    File_Adresses = [[f] for f in listdir(Files_Path) if isfile(join(Files_Path, f))]
    
    results = []
    
    for i in range(len(File_Adresses)):
    
        f = File_Adresses[i]
        
        print(f[0] ,";  ", i, " of ", len(File_Adresses))
        
        print('Read data')
        data = pd.read_csv(join(Files_Path,f[0]), delim_whitespace = delim_whitespace).values
        
        if(len(data[0])>=3):
            data_error = cp.copy(data[:,2])
        
        else:
            data_error = [0.2 for _ in data]    
            
        print("Shifting data")
        data = Shift_Data(data)

        print("Removing General trends")
        data = Remove_General_Trends(data, plot = Plot)
        
        print("Normalising amplitudes")
        data = Normalise_Amplitude(data, plot = Plot)

        # Plot new form of data;
        if(Plot == True):
            plt.scatter(data[:,0],data[:,1],s = 10)

            plt.show()

        print("Calculating period")
        P = Calc_Period(data[:,0],data[:,1],data_error)

        P_Error = Est_Period_Error(data, data_error, P, 0.9*P, 1.1*P, 100, 10, Print = Plot)
        
        results.append([f,P,P_Error])
        
    pd.DataFrame(results, columns=['name','Period','Period error']).to_csv(Output_File_Address, index = None)

# -- Main -------------------

data = pd.read_csv("Raw Data/Ogle/I/LMC562.02.33.dat",
                   delim_whitespace = True).values

Calc_Period_For_All("Sin_sqrx/", "LS_Sin_sqrx_01.csv", 10, 500, 10000, 100, 100, delim_whitespace = False)
