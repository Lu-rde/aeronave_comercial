import numpy as np
import scipy.linalg as la

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
