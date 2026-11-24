import numpy as np
import matplotlib.pyplot as plt

def my_regression(x, y, deg):
    m_plus_1 = len(x)
    n = deg
    A = np.array([[x[j]**k for k in range(n + 1)] for j in range(m_plus_1)])
    w = np.linalg.solve(A.T @ A, A.T @ y)
    return w

# Método manual con Vandermonde y Ecuaciones Normales
w_my = my_regression(x_ils, h_ils, deg=1)
w0_est, w1_est = w_my, w_my

# Método directo de NumPy
coef_numpy = np.polyfit(x_ils, h_ils, deg=1)
m_numpy, n_numpy = coef_numpy, coef_numpy

# Cálculo del ángulo de descenso gamma (en grados)
gamma_my = np.degrees(np.arctan(np.abs(w1_est)))
gamma_numpy = np.degrees(np.arctan(np.abs(m_numpy)))

# Evaluación del modelo
h_pred = w0_est + w1_est * x_ils

