import numpy as np
import scipy.linalg as la


def my_Gauss(A, b):
    """Resuelve el sistema lineal Ax = b mediante eliminación gaussiana con pivoteo parcial y sustitución regresiva."""
    # Convertimos A y b a float y copiamos para no modificar los arrays originales fuera de la función
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    N = len(b)  # Número de ecuaciones / incógnitas (sistema N x N)
    x = np.zeros(N, dtype=float)  # Vector solución inicializado a ceros

    # FASE 1: Eliminación hacia adelante (convertir A en matriz triangular superior)
    for i in range(N):
        # Buscamos el elemento de mayor valor absoluto en la columna i, desde la fila i hacia abajo,
        # para usar pivoteo parcial y reducir errores de redondeo o división por cero
        pivote = i + np.argmax(np.abs(A[i:, i]))
        if np.isclose(A[pivote, i], 0.0):
            raise ValueError("La matriz es singular o no tiene pivote no nulo.")
        # Intercambiamos la fila actual i con la fila del pivote (tanto en A como en b)
        if pivote != i:
            A[[i, pivote]] = A[[pivote, i]]
            b[[i, pivote]] = b[[pivote, i]]

        # Anulamos los elementos por debajo del pivote en la columna i
        for j in range(i + 1, N):
            factor = A[j, i] / A[i, i]  # Multiplicador de eliminación de la fila j
            A[j, i:] -= factor * A[i, i:]  # Actualizamos el resto de la fila j en A
            b[j] -= factor * b[i]  # Aplicamos la misma operación al término independiente b[j]

    # FASE 2: Sustitución hacia atrás (despejamos las incógnitas desde la última fila hasta la primera)
    for i in range(N - 1, -1, -1):
        # Restamos al término independiente b[i] la contribución de las incógnitas ya calculadas (x[i+1:])
        # y dividimos entre el elemento diagonal A[i, i]
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x


def Gauss_numpy(A, b):
    """Resuelve Ax = b utilizando el solver directo optimizado de NumPy (basado en LAPACK)."""
    return np.linalg.solve(A, b)


def LU(A, b):
    """Resuelve Ax = b factorizando A = P * L * U con SciPy y resolviendo los sistemas triangulares."""
    lu, piv = la.lu_factor(A)  # Calcula la descomposición LU compacta y los índices de pivoteo
    return la.lu_solve((lu, piv), b)  # Resuelve primero L y luego U para obtener x


def my_Jacobi(A, b, tol=1e-6, max_iter=500):
    """
    Resuelve Ax = b de forma iterativa mediante el método de Jacobi.
    Devuelve: (vector_solucion, iteraciones_realizadas, ha_convergido).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    N = len(b)
    x = np.zeros(N, dtype=float)  # Estimación inicial de la solución (vector nulo)
    diag = np.diag(A)  # Elementos de la diagonal principal de A

    # Un elemento diagonal nulo impediría dividir por A[i, i] en cada iteración
    if np.any(np.isclose(diag, 0.0)):
        raise ValueError("Jacobi requiere elementos diagonales distintos de cero.")

    # Matriz con los términos fuera de la diagonal (A sin su diagonal principal)
    R_off = A - np.diag(diag)

    for k in range(1, max_iter + 1):
        # En Jacobi, todas las componentes nuevas se calculan usando el vector x de la iteración anterior
        x_new = (b - R_off @ x) / diag

        # Comprobamos si aparecen NaN o infinitos (señal de divergencia numérica)
        if not np.all(np.isfinite(x_new)):
            return x_new, k, False

        # Criterio de parada: diferencia máxima entre iteraciones consecutivas menor que tol
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k, True

        x = x_new  # Actualizamos x para la siguiente iteración

    # Si se agotan max_iter sin cumplir tol, devolvemos el último vector y False en convergencia
    return x, max_iter, False


def my_Gauss_Seidel(A, b, tol=1e-6, max_iter=500):
    """
    Resuelve Ax = b de forma iterativa mediante el método de Gauss-Seidel.
    Devuelve: (vector_solucion, iteraciones_realizadas, ha_convergido).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    N = len(b)
    x = np.zeros(N, dtype=float)  # Estimación inicial de la solución

    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("Gauss-Seidel requiere elementos diagonales distintos de cero.")

    for k in range(1, max_iter + 1):
        x_old = x.copy()  # Guardamos una copia previa para evaluar el criterio de convergencia al final
        for i in range(N):
            # Componentes ya actualizadas en esta misma iteración (índices < i) -> usamos x[:i]
            suma_izq = np.dot(A[i, :i], x[:i])
            # Componentes aún no actualizadas en esta iteración (índices > i) -> usamos x_old[i+1:]
            suma_der = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            # Despejamos la variable x[i] usando los valores más recientes disponibles
            x[i] = (b[i] - suma_izq - suma_der) / A[i, i]

        if not np.all(np.isfinite(x)):
            return x, k, False

        if np.max(np.abs(x - x_old)) < tol:
            return x, k, True

    return x, max_iter, False


def resolver_con_consenso(A, b, tol_disp=1e-3, incluir_iterativos_si_convergen=True):
    """
    Calcula la solución de Ax = b con varios métodos, evalúa la dispersión máxima
    entre los resultados válidos y devuelve la media como solución final si está
    dentro del umbral tol_disp.
    """
    # Calculamos primero las soluciones de los métodos directos (exactos salvo redondeo)
    sol_gauss = my_Gauss(A, b)
    sol_numpy = Gauss_numpy(A, b)
    sol_lu = LU(A, b)

    # Diccionario con todos los métodos candidatos a entrar en el consenso
    candidatos = {
        "Gauss manual": sol_gauss,
        "Gauss numpy": sol_numpy,
        "LU scipy": sol_lu,
    }
    info_iter = {}  # Guardará (iteraciones, convergio) de los métodos iterativos

    # Evaluamos métodos iterativos y solo los añadimos al consenso si convergen y son finitos
    if incluir_iterativos_si_convergen:
        sol_jac, it_jac, conv_jac = my_Jacobi(A, b)
        sol_gs, it_gs, conv_gs = my_Gauss_Seidel(A, b)
        info_iter["Jacobi"] = (it_jac, conv_jac)
        info_iter["Gauss-Seidel"] = (it_gs, conv_gs)

        if conv_jac and np.all(np.isfinite(sol_jac)):
            candidatos["Jacobi"] = sol_jac
        if conv_gs and np.all(np.isfinite(sol_gs)):
            candidatos["Gauss-Seidel"] = sol_gs

    # Apilamos las soluciones candidatas: forma (num_metodos, N)
    matriz_sol = np.array(list(candidatos.values()))

    # Margen de dispersión absoluto: máxima diferencia (max - min) a lo largo de los métodos,
    # tomando el peor caso entre todas las incógnitas (axis=0 -> por componente, np.max -> global)
    dispersion_abs = float(np.max(np.ptp(matriz_sol, axis=0)))

    # Media aritmética de las soluciones candidatas para cada componente
    media = np.mean(matriz_sol, axis=0)

    # Escala de referencia para calcular la dispersión relativa (evitando división por cero)
    escala = max(float(np.max(np.abs(media))), 1e-12)
    dispersion_rel = dispersion_abs / escala

    # Comprobamos si la dispersión entre métodos está por debajo del margen permitido
    dentro_umbral = dispersion_abs <= tol_disp

    if dentro_umbral:
        x_final = media
        criterio = f"Media de {len(candidatos)} métodos (dispersión {dispersion_abs:.2e} <= {tol_disp:.2e})"
    else:
        # Plan B: si la dispersión supera el umbral, tomamos la media de los métodos directos
        x_final = np.mean([sol_gauss, sol_numpy, sol_lu], axis=0)
        criterio = (
            f"AVISO: dispersión {dispersion_abs:.2e} > {tol_disp:.2e}. "
            "Se usa media de métodos directos (Gauss/NumPy/LU)."
        )

    # Devolvemos un diccionario con el resultado final y toda la trazabilidad del proceso
    return {
        "x_final": x_final,
        "resultados": candidatos,
        "info_iter": info_iter,
        "dispersion_abs": dispersion_abs,
        "dispersion_rel": dispersion_rel,
        "dentro_umbral": dentro_umbral,
        "criterio": criterio,
    }

# APARTADO 2

def my_regression(x, y):
    """
    Ajuste lineal por mínimos cuadrados y = m*x + n mediante ecuaciones normales
    matriciales: theta = (X^T @ X)^(-1) @ (X^T @ y), donde X = [x, 1].
    Devuelve: (m, n)
    """
    x = np.asarray(x, dtype=float).ravel()  # Vector columna implícito de variable independiente (N,)
    y = np.asarray(y, dtype=float).ravel()  # Vector de variable dependiente (N,)

    # Construimos la matriz de diseño X de tamaño (N, 2): primera columna x, segunda columna de unos
    X = np.column_stack((x, np.ones_like(x)))

    # Ecuaciones normales: X^T @ X @ theta = X^T @ y -> resolvemos con np.linalg.solve (más estable que inv)
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.solve(XtX, Xty)

    m = float(theta[0])  # Pendiente de la recta ajustada
    n = float(theta[1])  # Ordenada en el origen
    return m, n


def regression_numpy(x, y):
    """
    Ajuste lineal por mínimos cuadrados y = m*x + n utilizando np.polyfit de grado 1.
    Devuelve: (m, n)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # np.polyfit(x, y, 1) devuelve [pendiente, ordenada_en_el_origen]
    m, n = np.polyfit(x, y, 1)
    return float(m), float(n)


def ajustar_senda_consenso(x, y, tol_disp=1e-9):
    """
    Calcula la regresión lineal con my_regression y regression_numpy, comprueba la
    dispersión entre ambos métodos y devuelve (m, n, gamma_deg, info).
    """
    m_my, n_my = my_regression(x, y)
    m_np, n_np = regression_numpy(x, y)

    # Dispersión máxima entre pendientes y ordenadas de ambos métodos
    disp_m = abs(m_my - m_np)
    disp_n = abs(n_my - n_np)
    dispersion = max(disp_m, disp_n)

    if dispersion <= tol_disp:
        m_final = 0.5 * (m_my + m_np)
        n_final = 0.5 * (n_my + n_np)
        dentro_umbral = True
    else:
        # Respaldo directo con el ajuste de NumPy si supera la tolerancia
        m_final, n_final = m_np, n_np
        dentro_umbral = False

    # Ángulo de descenso en grados: gamma = arctan(m)
    gamma_deg = float(np.degrees(np.arctan(m_final)))

    info = {
        "m_my_regression": m_my,
        "n_my_regression": n_my,
        "m_numpy": m_np,
        "n_numpy": n_np,
        "dispersion": dispersion,
        "dentro_umbral": dentro_umbral,
    }
    return m_final, n_final, gamma_deg, info