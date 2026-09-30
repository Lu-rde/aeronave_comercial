import numpy as np


def resolver_reparto_cargas(params_masa, matriz_geometria):
    """Apartado 1: Resuelve A * x = b (reacciones Rm, Rpi, Rpd en Newtons)."""
    # params_masa[1:] contiene el peso total y los momentos en X e Y
    return np.linalg.solve(matriz_geometria, params_masa[1:])


def calcular_senda_ils(x_ils, h_ils):
    """Apartado 2: Ecuaciones normales (X^T * X)^(-1) * X^T * y para la senda ILS."""
    X = np.column_stack((x_ils, np.ones_like(x_ils)))
    m, n = np.linalg.inv(X.T @ X) @ X.T @ h_ils
    gamma_deg = np.degrees(np.arctan(np.abs(m)))
    return m, n, gamma_deg