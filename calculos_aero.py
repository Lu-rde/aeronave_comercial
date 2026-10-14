import numpy as np


def resolver_reparto_cargas(params_masa, matriz_geometria):
    """Apartado 1: Resuelve A * x = b para obtener las reacciones [Rm, Rpi, Rpd] en Newtons."""
    # params_masa[1:] contiene [Peso Total, Momento X, Momento Y]
    return np.linalg.solve(matriz_geometria, params_masa[1:])


def calcular_senda_ils(x_ils, h_ils):
    """Apartado 2: Ecuaciones Normales (X^T * X)^(-1) * X^T * y para la regresión de la senda ILS."""
    X = np.column_stack((x_ils, np.ones_like(x_ils)))
    m, n = np.linalg.inv(X.T @ X) @ X.T @ h_ils
    gamma_deg = np.degrees(np.arctan(np.abs(m)))
    return m, n, gamma_deg


def calcular_viento_decision(altitudes, velocidades, altitud_decision=60.0):
    """Apartado 3: Interpolación 1D del viento a la Altitud de Decisión (60m)."""
    return float(np.interp(altitud_decision, altitudes, velocidades))


def calcular_tiempo_taxi(t0=10.0, v_target=15.0, tol=1e-6, max_iter=100):
    """Apartado 4: Método de Newton-Raphson para f(t) = V(t) - 15 = 0.

    Único bucle while permitido por la guía docente.
    """
    t = float(t0)
    f = 70.0 * np.exp(-0.04 * t) - 1.2 * t - v_target
    iteraciones = 0
    while abs(f) > tol and iteraciones < max_iter:
        df = -2.8 * np.exp(-0.04 * t) - 1.2
        t -= f / df
        f = 70.0 * np.exp(-0.04 * t) - 1.2 * t - v_target
        iteraciones += 1
    return t


def calcular_desaceleracion(tiempo, velocidad):
    """Apartado 5: Diferencias finitas centrales vectorizadas mediante slicing de NumPy."""
    aceleracion = np.empty_like(velocidad)
    # Diferencias centrales para puntos interiores
    aceleracion[1:-1] = (velocidad[2:] - velocidad[:-2]) / (
        tiempo[2:] - tiempo[:-2]
    )
    # Extremos (diferencias progresivas/regresivas)
    aceleracion = (velocidad[14] - velocidad) / (tiempo[14] - tiempo)
    aceleracion[-1] = (velocidad[-1] - velocidad[-2]) / (
        tiempo[-1] - tiempo[-2]
    )
    return aceleracion


def calcular_distancia_frenado(tiempo, velocidad, factor_masa=1.0):
    """Apartado 6: Integración numérica vectorial por Regla del Trapecio ajustada por masa."""
    return float(np.trapz(velocidad * factor_masa, tiempo))