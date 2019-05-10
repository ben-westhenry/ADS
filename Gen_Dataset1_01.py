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

def Gen_Sinx(Sample_times, P, A, noise, Flux_offset, Phase, RW_A):

    f = [Flux_offset for i in range(len(Sample_times))]
    
    f += Rand_Walk(Sample_times, RW_A)
    
    for i in range(len(Sample_times)):
        f[i] += A*np.sin(2*np.pi*(Sample_times[i]-Phase)/P)
        
    f += np.random.normal(loc = 0, scale = noise, size = len(f))
        
    return f
    
def Gen_Sin_sqrx(Sample_times, P, A, noise, Flux_offset, Phase, RW_A):
    
    f = [Flux_offset for i in range(len(Sample_times))]
    
    f += Rand_Walk(Sample_times, RW_A)
    
    for i in range(len(Sample_times)):
        x = (Sample_times[i] - Phase)%P
        f[i] += A*np.sin(2*np.pi*(x/P)**2)
        
    f += np.random.normal(loc = 0, scale = noise, size = len(f))
        
    return f
    
def Gen_Non_Periodic(Sample_times, noise, Flux_offset, RW_A):

    f = [Flux_offset for i in range(len(Sample_times))]
    
    f += Rand_Walk(Sample_times, RW_A)
    
    f += np.random.normal(loc = 0, scale = noise, size = len(f))
    
    return f
    
# Read out data
data = pd.read_csv("Raw Data/Ogle/I/LMC562.01.11.dat", delim_whitespace = True).values
    
sample_times = data[:,0]

Meta_Data = []

for i in range(500):

    print('Sinx Iteration; ' + str(i))
    
    # Randomly generate paramiters;
    
    P = np.random.uniform(0.1,400)
    
    A = np.random.uniform(0.5,4)
    
    noise = np.random.uniform(0.1,0.4)
    
    Flux_offset = np.random.uniform(8,14)
    
    Phase = np.random.uniform(0,P)
    
    file_name = 'Sinx' + format(i, '03') + '.csv'
    
    # Add paramiters to meta data
    Meta_Data.append([file_name, P, A, noise, Flux_offset, Phase])
    
    # Gen flux values
    f = Gen_Sinx(sample_times, P, A, noise, Flux_offset, Phase, 0.01)
    
    # Save data
    df = pd.DataFrame({'Time':sample_times, 'Flux':f})
    df.to_csv(path_or_buf = 'Sinx/'+file_name, index = False)

df = pd.DataFrame(Meta_Data, columns = ['file_name', 'P', 'A', 'noise', 'Flux_offset', 'Phase'])
df.to_csv(path_or_buf = 'Sinx_Meta_data.csv', index = False)

Meta_Data = []

for i in range(500):

    print('Sin sqrx Iteration; ' + str(i))
    
    # Randomly generate paramiters;
    
    P = np.random.uniform(0.1,400)
    
    A = np.random.uniform(0.5,4)
    
    noise = np.random.uniform(0.1,0.4)
    
    Flux_offset = np.random.uniform(8,14)
    
    Phase = np.random.uniform(0,P)
    
    file_name = 'Sin_sqrx' + format(i, '03') + '.csv'
    
    # Add paramiters to meta data
    Meta_Data.append([file_name, P, A, noise, Flux_offset, Phase])
    
    f = Gen_Sin_sqrx(sample_times, P, A, noise, Flux_offset, Phase, 0.01)

    df = pd.DataFrame({'Time':sample_times, 'Flux':f})
    
    df.to_csv(path_or_buf = 'Sin_sqrx/'+file_name, index = False)
    
df = pd.DataFrame(Meta_Data, columns = ['file_name', 'P', 'A', 'noise', 'Flux_offset', 'Phase'])
df.to_csv(path_or_buf = 'Sin_sqrx_Meta_data.csv', index = False)

Meta_Data = []

for i in range(100):

    print('Non periodic Iteration; ' + str(i))
    
    # Randomly generate paramiters;
    
    noise = np.random.uniform(0.1,0.4)
    
    Flux_offset = np.random.uniform(8,14)
    
    file_name = 'Non_Periodic' + format(i, '03') + '.csv'
    
    # Add paramiters to meta data
    Meta_Data.append([file_name, noise, Flux_offset])
    
    f = Gen_Non_Periodic(sample_times, noise, Flux_offset, 0.1)

    df = pd.DataFrame({'Time':sample_times, 'Flux':f})
    
    
    df.to_csv(path_or_buf = 'Non_Periodic/'+file_name, index = False)
    
df = pd.DataFrame(Meta_Data, columns = ['file_name', 'noise', 'Flux_offset'])
df.to_csv(path_or_buf = 'Non_Periodic_Meta_data.csv', index = False)
