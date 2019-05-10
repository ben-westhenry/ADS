import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
from os import listdir
from os.path import isfile, join

def Plot_N_Save(File_Address, Save_Address, Title):
    
    print("Loading; " + File_Address)
    
    data = pd.read_csv(File_Address, delim_whitespace = True).values
    
    plt.figure(figsize=(20,10))
    
    plt.scatter(data[:,0],data[:,1])
    
    plt.title(Title)
    
    print("Saving to; " + Save_Address + ".png")
    
    plt.savefig(Save_Address + ".png")
    
    plt.clf()

def Plot_N_Save_Files(Files_Path, Save_Path, N_Start, N_End):
    
    File_Adresses = [[f] for f in listdir(Files_Path) if isfile(join(Files_Path, f))]
    
    File_Adresses = File_Adresses[N_Start:N_End]
    
    for f in File_Adresses:
        
        Plot_N_Save(join(Files_Path,f[0]), join(Save_Path,f[0][:-4]), f[0])

Plot_N_Save_Files("Raw Data/Ogle/I/","Raw Data/Ogle_Plots", 0,1000)
