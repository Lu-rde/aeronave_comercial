import numpy as np 
import matplotlib.pyplot as plt 
# ------------------------------------------------------------------------------ 
# # 1\. DATOS DEL PROBLEMA 
# # ------------------------------------------------------------------------------ 
# # Vector de tiempo de 0 a 35 segundos (1000 puntos) 
tiempo = np.linspace(0, 35, 1000) 
# Perfil de velocidad del avión [m/s] 
velocidad = 70.0 * np.exp(-0.04 * tiempo) - 1.2 * tiempo
velocidad[velocidad < 0] = 0.0 # El avión no marcha atrás
# ------------------------------------------------------------------------------
# 2\. MÉTODO 1: my_3PTOS_centradas (Fórmula de 3 puntos centrada)
# ------------------------------------------------------------------------------
def my_3PTOS_centradas(v, t):
    n = len(v)
    dt = t[1] - t[0]
    a = np.zeros(n)
    # Extremo inicial: 3 puntos adelantada
    a[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dt)
    # Puntos interiores: 3 puntos centrada O(dt^2)
    for i in range(1, n - 1):
        a[i] = (v[i + 1] - v[i - 1]) / (2.0 * dt)
    # Extremo final: 3 puntos retrasada
    a[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dt)
    return a
# ------------------------------------------------------------------------------
# 3\. MÉTODO 2: my_3PTOS_retrasadas (Fórmula de 3 puntos retrasada)
# ------------------------------------------------------------------------------
def my_3PTOS_retrasadas(v, t):
    n = len(v)
    dt = t[1] - t[0]
    a = np.zeros(n)
    # Ajuste de bordes iniciales para no salir de rango
    a[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dt)
    a[1] = (v[2] - v[0]) / (2.0 * dt)
    # Puntos desde i=2 hasta el final: 3 puntos retrasada O(dt^2)
    for i in range(2, n):
        a[i] = (3.0 * v[i] - 4.0 * v[i - 1] + v[i - 2]) / (2.0 * dt)
    return a
# ------------------------------------------------------------------------------
# 4\. CÁLCULO Y COMPARACIÓN DE LOS 3 MÉTODOS
# ------------------------------------------------------------------------------
# A) Método 1: my_3PTOS centradas
a_centrada = my_3PTOS_centradas(velocidad, tiempo)
# B) Método 2: my_3PTOS retrasadas
a_retrasada = my_3PTOS_retrasadas(velocidad, tiempo)
# C) Método 3: NumPy (np.gradient)
a_numpy = np.gradient(velocidad, tiempo)
# D) Solución analítica exacta (para validar)
a_analitica = -2.8 * np.exp(-0.04 * tiempo) - 1.2
# ------------------------------------------------------------------------------
# 5\. IMPRESIÓN DE RESULTADOS POR CONSOLA
# ------------------------------------------------------------------------------
print("=============================================================")
print("--- APARTADO 5: DERIVACIÓN NUMÉRICA (DESACELERACIÓN) ---")
print("=============================================================")
print("\\n1\. En t = 0.00 s (Momento del impacto / Toma):")
print(f" my_3PTOS centradas; Aceleración: {a_centrada[0]:.2f} m/s²")
print(f" my_3PTOS retrasadas; Aceleración: {a_retrasada[0]:.2f} m/s²")
print(f" np.gradient (NumPy); Aceleración: {a_numpy[0]:.2f} m/s²")
print(f" Analítica exacta; Aceleración: {a_analitica[0]:.2f} m/s²")
# Instante de velocidad de rodaje (t = 22.45 s)
idx_taxi = np.argmin(np.abs(tiempo - 22.45))
print(f"\\n2\. En t = 22.45 s (Velocidad de Rodaje V\_taxi = 15 m/s):")
print(f" my_3PTOS centradas; Aceleración: {a_centrada[idx_taxi]:.2f} m/s²")
print(f" my_3PTOS retrasadas; Aceleración: {a_retrasada[idx_taxi]:.2f} m/s²")
print(f" np.gradient (NumPy); Aceleración: {a_numpy[idx_taxi]:.2f} m/s²")
print(f" Analítica exacta; Aceleración: {a_analitica[idx_taxi]:.2f} m/s²")