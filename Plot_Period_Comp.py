import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
from itertools import cycle

def Rm_Pading(data, Fp, Bp):
    
    for i in range(len(data)):
        
        data[i,0] = data[i,0][Fp:-Bp]
        
def Match(data_a, data_b):

    result = []

    for a_val in data_a:
        for b_val in data_b:
            if(a_val[0] == b_val[0]):
                tmp = []
                for a in a_val:
                    tmp.append(a)
                for b in b_val[1:]:
                    tmp.append(b)
                
                result.append(tmp)
                break
    return result
    
def Plot_Set(Ogle_Data, Data):

    Rm_Pading(Data, 2, 6)
    
    print(Data)
    
    print(Ogle_Data)
    


    combined = Match(Ogle_Data, Data)

    combined = np.asarray(combined)

    print(combined)

    # --- Plot calculated period against true period ------------------------

    plt.scatter([float(i) for i in combined[:,1]],[float(i) for i in combined[:,2]],s=1, c = 'black')
    plt.xlabel('Ogle period')
    plt.ylabel('Calculated period')

    plt.plot([0,700],[0,3*700  ], label='3:1')
    plt.plot([0,700],[0,2*700  ], label='2:1')
    plt.plot([0,700],[0,  700  ], label='1:1')
    plt.plot([0,700],[0,  700/2], label='1:2')
    plt.plot([0,700],[0,  700/3], label='1:3')
    plt.legend()

    plt.xlim(0,700)
    plt.ylim(0,700)

    plt.show()

    # --- Plot difference in histogram ---------------------------------------


    difference = np.asarray([float(i) for i in combined[:,2]]) - np.asarray([float(i) for i in combined[:,1]])

    print('Mean difference; ', np.mean(difference), ' Standard Deviation; ', np.std(difference))

    plt.hist(difference,bins = 100)
    plt.xlabel('Calculated period - Ogle period')
    plt.show()




Ogle_Raw_Lines = pd.read_csv("Raw Data/list.dat").values

Ogle_data = []

#LMC563.16.842    05:46:55.53 -66:30:47.6 OTHER        18.925 18.981    1.85094   0.069
#012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
#0         1         2         3         4         5         6         7         8

for val in Ogle_Raw_Lines:
    
    tmp = cp.copy(val[0][68:78])
    
    tmp.strip()
    
    if(len(tmp) <= 0):
        print(val[0][:17],"Missing period")

    else:
        Ogle_data.append([val[0][:17].strip(), float(val[0][68:78])])
    
    
Ogle_data = np.asarray(Ogle_data)

Data = pd.read_csv("LSPeriods_02.csv").values

Plot_Set(Ogle_data, Data)

