import numpy as np
from utils import _A, _B, _C, _D

def ackley(x):
    sum1 = 0
    sum2 = 0
    for i in range(_D):
        sum1 += x[i]**2
        sum2 += np.cos(_C*x[i])

    fitness = -_A*np.exp(-_B * np.sqrt(sum1/_D)) - np.exp(sum2/_D) + _A + np.exp(1)

    return fitness

def schaffer(x):     

    f6 = 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5)/(1 + 0.001*(x[0]**2 + x[1]**2))**2
    
    return f6

def rastrigin(x):
    component = 0
    for i in range(20):
        component += (pow(x[i],2)-10*np.cos(_C*x[i]))
    f7 = 200 + component
    
    return f7