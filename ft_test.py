import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Xk(xn,pn,wk):
    
    a = [np.sum(xn*np.e**(-2*np.pi*1j*pn*wk[i])) for i in range(len(wk))]
    
    return a

def xn(Xk,pn,wk):
    
    return [np.sum(Xk*np.e**( 2*np.pi*1j*pn[i]*wk))         for i in range(len(pn))]


#data = pd.read_csv("Raw Data/Ogle/I/LMC562.01.2.dat", delim_whitespace = True).values


tmp = np.asarray([ [t,0.5*np.cos(5*t)+1*np.cos(8*t)] for t in np.random.uniform(0,10,1000) ])

data = []

for a in tmp:
    if a[0] < 3 or a[0] > 5:
        data.append(a)
        
data.sort(key = lambda tup: tup[0])

data = np.asarray(data)

plt.scatter(data[:,0], data[:,1], label = "Original")

plt.show()

count = 100000

wk = np.asarray([10*i/count for i in range(0,count)])

a = Xk(data[:,1],data[:,0],wk)

plt.plot(wk,a)
plt.show()


