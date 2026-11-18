import numpy as np
import scipy.linalg as la

# ==============================================================================
# CARGA DE DATOS DE TELEMETRÍA (TEMA 4)
# ==============================================================================
# Datos para el Sistema Lineal (Geometria del tren y pesos)
# x_cg, peso_total, momentos en x e y
params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0]) 
matriz_geometria = np.array([
		[1.0, 1.0, 1.0],           # Suma de fuerzas verticales
		[3.2, 20.1, 20.1],         # Brazos de palanca eje X 
		[0.0, -4.5, 4.5]           # Brazos de palanca eje Y
		])

# Vector de cargas estáticas b: [Peso\_total, Momento\_X, Momento\_Y] 
b = params_masa[1:] 
# Extrae [735750.0, 13600000.0, 0.0]
		
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

#==============================
# 1. METODO DE GAUSS
#==============================
# a) Función Propia
def my_Gauss(A,b):
    N=len(b)
    A = A.astype(float).copy() 
    b = b.astype(float).copy()
    x=np.zeros(N)
    for i in range(N):
        for j in range(i+1,N):
            C=A[j,i]/A[i,i]
            A[j,:] = A[j,:] - C*A[i,:]
            b[j] = b[j] - C*b[i]
    for i in range(N-1,-1,-1):
        x[i] = (b[i] - np.dot(A[i,i+1:N],x[i+1:N]))/A[i,i]
    return x

# b) Función de Numpy
def Gauss_numpy(A,b):
    return np.linalg.solve(A,b)

#==============================
# 2. METODO DE LU
#==============================
def LU(A,b):
    lu, piv = la.lu_factor(A)
    x = la.lu_solve((lu, piv), b)
    return x
     
#=============================
# 3. METODO DE JACOBI
#=============================
def my_Jacobi(A,b, tol=1e-6, max_iter=500):
    N=len(b)
    x,y,R=np.zeros(N),np.zeros(N),np.zeros(N)
    LU=np.zeros((N,N))
    LU[:,:] = A[:,:]
    for i in range(N):
        LU[i,i] = 0
    for _ in range(max_iter):
        R=b-np.matmul(LU,x)
        for i in range(N):
            y[i]=R[i]/A[i,i]
        if np.max(np.abs(y - x)) < tol: 
            break 
        x = y.copy()
    return y

#=============================
# 4. METODO DE GAUSS-SEIDEL
#============================
def my_Gauss_Seidel(A,b, tol=1e-6, max_iter=500):
    N=len(b)
    x,R=np.zeros(N),np.zeros(N)
    for _ in range(max_iter): 
        x_old = x.copy()
        for i in range(N):
            R[i]=b[i]-np.dot(A[i,i+1:],x[i+1:])
        for i in range(N):
            x[i]=(R[i]-np.dot(A[i,:i],x[:i]))/A[i,i]
        if np.max(np.abs(x - x_old)) < tol: 
            break
    return x
