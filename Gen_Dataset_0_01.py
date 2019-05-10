import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd

def Rand_Walk(Sample_times, A):

    y = 0
    result = []
    result.append(y)
    for i in range(1,len(Sample_times)):
        y += np.random.normal(scale=A*(Sample_times[i]-Sample_times[i-1]))
        result.append(y)
        
    return np.array(result)

def Gen_Sinx(Sample_times, P, A, noise, Flux_offset, Phase):

    f = [Flux_offset for i in range(len(Sample_times))]
    
    f += Rand_Walk(Sample_times, 8)
    
    for i in range(len(Sample_times)):
        f[i] += A*np.sin(2*np.pi*(Sample_times[i]-Phase)/P)
        
    f += np.random.normal(loc = 0, scale = noise, size = len(f))
        
    return f
    
def Gen_Sin_sqrx(Sample_times, P, A, noise, Flux_offset, Phase):
    
    f = [Flux_offset for i in range(len(Sample_times))]
    
    f += Rand_Walk(Sample_times, 8)
    
    for i in range(len(Sample_times)):
        x = (Sample_times[i] - Phase)%P
        f[i] += A*np.sin(2*np.pi*(x/P)**2)
        
    f += np.random.normal(loc = 0, scale = noise, size = len(f))
        
    return f
    
sample_times = [np.random.uniform(0,10) for i in range(1000)]

sample_times.sort()

f = Gen_Sinx(sample_times, 3, 1, 0.3, 10, 0.1)

plt.scatter(sample_times, f, s = 1)

plt.show()

f = Gen_Sin_sqrx(sample_times, 3, 1, 0.3, 10, 0.1)

plt.scatter(sample_times, f, s = 1)

plt.show()
