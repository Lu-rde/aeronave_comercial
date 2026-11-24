import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import lagrange 

# ------------------------------------------------------------------------------ 
# 1. FUNCIÓN PROPIA LAGRANGE
# ------------------------------------------------------------------------------ 
def my_lagrange(x, y, punto): 
    n = len(x) 
    resultado = 0.0 
    for i in range(n): 
        L = 1.0 
        for j in range(n): 
            if i != j:
                L = L * (punto - x[j]) / (x[i] - x[j]) 
        resultado = resultado + y[i] * L 
    return resultado


