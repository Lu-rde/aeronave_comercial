import numpy as np
from math_utils import resolver_con_consenso

# ==============================================================================
# CARGA DE DATOS DE TELEMETRÍA (TEMA 4)
# ==============================================================================
# Parámetros de masa y momentos: [x_cg (m), peso_total (N), momento_X (N·m), momento_Y (N·m)]
params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0])

# Matriz de coeficientes del sistema lineal para los 3 apoyos del tren de aterrizaje:
# columnas -> [tren_morro, tren_principal_izq, tren_principal_der]
matriz_geometria = np.array([
    [1.0, 1.0, 1.0],      # Suma de fuerzas verticales = peso total
    [3.2, 20.1, 20.1],    # Brazos de palanca en eje X (momentos de cabeceo)
    [0.0, -4.5, 4.5],     # Brazos de palanca en eje Y (momentos de alabeo)
])

# Datos para Regresión Lineal (Telemetría ILS: distancia a pista vs. altitud con ruido)
x_ils = np.linspace(-10000, -500, 200)
h_ils = 0.0524 * np.abs(x_ils) + 15.0 + np.random.normal(0, 8, 200)

# Datos para Interpolación 1D (perfil vertical de viento en la capa límite)
altitudes_viento = np.array([0, 50, 100, 200, 500, 1000])  # [m]
velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3])  # [m/s]

# Datos para Derivación e Integración (perfil de velocidad durante el frenado)
tiempo_frenado = np.linspace(0, 35, 1000)  # [s]
velocidad_frenado = 70.0 * np.exp(-0.04 * tiempo_frenado) - 1.2 * tiempo_frenado  # [m/s]
velocidad_frenado[velocidad_frenado < 0] = 0

# Vector de cargas estáticas b: [Peso_total, Momento_X, Momento_Y]
b = params_masa[1:]

# ==============================================================================
# CÁLCULO INTERNO (MÉTODOS, CONVERGENCIA Y MARGEN DE DISPERSIÓN)
# ==============================================================================
# Todo el cálculo, control de convergencia y comprobación de dispersión ocurre aquí dentro
reporte = resolver_con_consenso(matriz_geometria, b, tol_disp=1e-2)
x = reporte["x_final"]  # Solución final [R_m, R_pi, R_pd] lista para cálculos posteriores

# ==============================================================================
# SALIDA LIMPIA
# ==============================================================================
print("=============================================================")
print("--- FUERZAS DE REACCIÓN EN LOS TRENES DE ATERRIZAJE ---")
print("=============================================================")
print(f"Tren de Morro (R_m):              {x[0]/1000:.2f} kN ({x[0]:.2f} N)")
print(f"Tren Principal Izquierdo (R_pi): {x[1]/1000:.2f} kN ({x[1]:.2f} N)")
print(f"Tren Principal Derecho (R_pd):   {x[2]/1000:.2f} kN ({x[2]:.2f} N)")