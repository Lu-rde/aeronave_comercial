from ecuaciones_lineales import *
from regresion_lineal import *
from interpolación_1D import *
from scipy.interpolate import lagrange 

# ==============================================================================
# CARGA DE DATOS DE TELEMETRÍA (
# ==============================================================================
# Datos para el Sistema Lineal (Geometria del tren y pesos)
# x_cg, peso_total, momentos en x e y
params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0]) 
matriz_geometria = np.array([
		[1.0, 1.0, 1.0],           # Suma de fuerzas verticales
		[3.2, 20.1, 20.1],         # Brazos de palanca eje X 
		[0.0, -4.5, 4.5]           # Brazos de palanca eje Y
		])
		
# Datos para Regresion Lineal (Telemetria ILS: Distancia a pista vs Altitud)
x_ils = np.linspace(-10000, -500, 200)
# Altitud con ruido de medicion del radar
h_ils = 0.0524 * np.abs(x_ils) + 15.0 + np.random.normal(0, 8, 200) 
		
# Datos para Interpolacion 1D (Perfil de viento en la capa limite)
altitudes_viento = np.array([0, 50, 100, 200, 500, 1000]) # [m]
velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3]) # [m/s]
		
# Datos para Derivacion e Integracion (Dinamica de frenado)
tiempo_frenado = np.linspace(0, 35, 1000) # [s]
# Perfil de velocidad [m/s]
velocidad_frenado = 70.0 * np.exp(-0.04 * tiempo_frenado)-1.2*tiempo_frenado 
velocidad_frenado[velocidad_frenado < 0] = 0

#==============================================================================
# DATO BASE
#==============================================================================
# Vector de cargas estáticas b: [Peso\_total, Momento\_X, Momento\_Y] 
b = params_masa[1:] 
# Extrae [735750.0, 13600000.0, 0.0]

# Altitud de decisión donde evaluar el viento [m] 
h_buscada = 60.0 

#==============================================================================
# RESULTADOS APARTADO 1: SISTEMA LINEAL (REPARTO DE CARGAS)
#==============================================================================

print("1a. Gauss manual: ", my_Gauss(matriz_geometria, b)) 
print("1b. Gauss numpy: ", Gauss_numpy(matriz_geometria, b)) 
print("2. LU scipy: ", LU(matriz_geometria, b)) 
print("3. Jacobi manual: ", my_Jacobi(matriz_geometria, b)) 
print("4. Gauss-Seidel: ", my_Gauss_Seidel(matriz_geometria, b))
x = Gauss_numpy(matriz_geometria, b)
print("=============================================================")
print("--- FUERZAS DE REACCIÓN EN LOS TRENES DE ATERRIZAJE ---")
print("=============================================================")
print(f"Tren de Morro (R_m): {x[0]/1000:.2f} kN ({x[0]:.2f} N)") 
print(f"Tren Principal Izquierdo (R_pi): {x[1]/1000:.2f} kN ({x[1]:.2f} N)") 
print(f"Tren Principal Derecho (R_pd): {x[2]/1000:.2f} kN ({x[2]:.2f} N)")

#==============================================================================
# RESULTADOS APARTADO 2: REGRESION LINEAL (TELEMETRIA)
#==============================================================================

#==============================================================================
# RESULTADOS APARTADO 3: INTERPOLACION 1D (PERFIL DEVIENTO)
#==============================================================================
# CÁLCULO Y COMPARACIÓN DE RESULTADOS 

# A) Método propio de Lagrange 
my_viento = my_lagrange(altitudes_viento, velocidad_viento, h_buscada) 

# B) Método nativo de SciPy (scipy.interpolate.lagrange) 
polinomio_scipy = lagrange(altitudes_viento, velocidad_viento) 
viento_scipy = polinomio_scipy(h_buscada) 

# C) Método nativo de NumPy (Interpolación lineal local) 
my_viento_numpy = np.interp(h_buscada, altitudes_viento, velocidad_viento) 

print("=============================================================")
print("--- RESULTADOS DE LA INTERPOLACIÓN 1D (PERFIL DE VIENTO) ---") 
print("=============================================================")
print(f"Altitud de decisión {h_buscada:.2f} m")
print(f"my_lagrange  {my_viento:.2f} m/s") 
print(f"SciPy        {viento_scipy:.2f} m/s") 
print(f"NumPy interp {my_viento_numpy:.2f} m/s")