import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
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
    
def Plot_Set(Truth_Adress, FFT_Adress, PDM_Adress, LS_Adress):

    Sinx_Truth = pd.read_csv(Truth_Adress).values
    Sinx_FFT   = pd.read_csv(FFT_Adress  ).values
    Sinx_PDM   = pd.read_csv(PDM_Adress  ).values
    Sinx_LS    = pd.read_csv(LS_Adress   ).values


    Rm_Pading(Sinx_FFT, 2, 2)

    Rm_Pading(Sinx_PDM, 2, 2)

    Rm_Pading(Sinx_LS , 2, 2)

    combined = Match(Sinx_Truth, Sinx_FFT)
    combined = Match(combined, Sinx_PDM)
    combined = Match(combined, Sinx_LS)

    combined = np.asarray(combined)


    # --- Plot calculated period against true period ------------------------

    # FFT

    plt.scatter([float(i) for i in combined[:,1]],[float(i) for i in combined[:,6]],s=1, c = 'black')
    plt.xlabel('True period')
    plt.ylabel('FFT period')

    plt.plot([0,700],[0,3*700  ], label='3:1')
    plt.plot([0,700],[0,2*700  ], label='2:1')
    plt.plot([0,700],[0,  700  ], label='1:1')
    plt.plot([0,700],[0,  700/2], label='1:2')
    plt.plot([0,700],[0,  700/3], label='1:3')
    plt.legend()

    plt.xlim(0,700)
    plt.ylim(0,700)

    plt.show()

    # PDM

    plt.scatter([float(i) for i in combined[:,1]],[float(i) for i in combined[:,8]],s=1, c = 'black')
    plt.xlabel('True period')
    plt.ylabel('PDM period')

    plt.plot([0,700],[0,3*700  ], label='3:1')
    plt.plot([0,700],[0,2*700  ], label='2:1')
    plt.plot([0,700],[0,  700  ], label='1:1')
    plt.plot([0,700],[0,  700/2], label='1:2')
    plt.plot([0,700],[0,  700/3], label='1:3')
    plt.legend()

    plt.xlim(0,700)
    plt.ylim(0,700)

    plt.show()

    # LS

    plt.scatter([float(i) for i in combined[:,1]],[float(i) for i in combined[:,10]],s=1, c = 'black')
    plt.xlabel('True period')
    plt.ylabel('Lomscagle period')

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

    # FFT
    difference = np.asarray([float(i) for i in combined[:,6]]) - np.asarray([float(i) for i in combined[:,1]])

    print('FFT: Mean difference; ', np.mean(difference), ' Standard Deviation; ', np.std(difference))

    plt.hist(difference,bins = 100)
    plt.xlabel('Fourier transform period - Ogle period')
    plt.show()

    # PDM
    difference = np.asarray([float(i) for i in combined[:,8]]) - np.asarray([float(i) for i in combined[:,1]])

    print('PDM: Mean difference; ', np.mean(difference), ' Standard Deviation; ', np.std(difference))

    plt.hist(difference,bins = 100)
    plt.xlabel('Probability density measure period - Ogle period')
    plt.show()

    # LS
    difference = np.asarray([float(i) for i in combined[:,10]]) - np.asarray([float(i) for i in combined[:,1]])

    print('LS: Mean difference; ', np.mean(difference), ' Standard Deviation; ', np.std(difference))

    plt.hist(difference,bins = 100)
    plt.xlabel('Lomscagle period - Ogle period')
    plt.show()

print('----     Sin x    ----')

Plot_Set('Sinx_Meta_data.csv', 'FFT_Sin_01.csv', 'PDM_Sinx_02.csv', 'LS_Sinx01.csv')

print('----    Sin x^2   ----')

Plot_Set('Sin_sqrx_Meta_data.csv', 'FFT_Sin_sqrx01.csv', 'PDM_Sin_sqrx_02.csv', 'LS_Sin_sqrx_01.csv')

print('---- Non periodic ----')


Truth = pd.read_csv('Non_Periodic_Meta_data.csv').values
FFT   = pd.read_csv('FFT_Non_Periodic_01.csv'   ).values
PDM   = pd.read_csv('PDM_Non_Periodic_02.csv'   ).values
LS    = pd.read_csv('LS_Non_Periodic_01.csv'    ).values

# --- FFT ---

FFT = np.asarray([float(i) for i in FFT[:,1]])

plt.hist(FFT,bins = 10)
plt.xlabel('FFT calculated period for non-periodic data')
plt.show()

print('FFT: Mean; ', np.mean(FFT), ' Standard Deviation; ', np.std(FFT))

# --- PDM ---

PDM = np.asarray([float(i) for i in PDM[:,1]])

plt.hist(PDM,bins = 10)
plt.xlabel('PDM calculated period for non-periodic data')
plt.show()

print('PDM: Mean; ', np.mean(PDM), ' Standard Deviation; ', np.std(PDM))

# --- LS ---

LS = np.asarray([float(i) for i in LS[:,1]])

plt.hist(LS,bins = 10)
plt.xlabel('LS calculated period for non-periodic data')
plt.show()

print('LS: Mean; ', np.mean(LS), ' Standard Deviation; ', np.std(LS))






