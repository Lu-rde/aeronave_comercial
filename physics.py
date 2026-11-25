import numpy as np 
from math_utils import interpolar_con_consenso
from math_utils import derivar_con_consenso
from math_utils import integrar_con_consenso

def calcular_viento_altura_decision(altitudes_viento, velocidad_viento, altitud_decision=60.0):
    """
    Determina la velocidad del viento cruzado en la Altura de Decisión (60 m por defecto)[cite: 12]
    a partir del perfil discreto de la capa límite atmosférica[cite: 11].
    """
    v_dh, info_viento = interpolar_con_consenso(
        altitudes_viento,
        velocidad_viento,
        altitud_decision,
    )
    return v_dh, info_viento

def calcular_distancia_frenado(tiempo_frenado, velocidad_frenado, tol_disp=1e-2):
    """
    Calcula la distancia total de pista consumida S = integral(V(t) dt) [m]
    desde el touchdown hasta la detención de la aeronave[cite: 11, 12].
    """
    distancia_m, info_integral = integrar_con_consenso(
        velocidad_frenado,
        tiempo_frenado,
        tol_disp=tol_disp,
    )
    return distancia_m, info_integral



def calcular_perfil_desaceleracion(tiempo_frenado, velocidad_frenado, g=9.81, tol_disp=1e-2):
    """
    Calcula la aceleración instantánea a(t) = dV/dt [m/s^2] durante el frenado,
    la desaceleración máxima alcanzada y comprueba el límite de confort (0.3g ~ 2.94 m/s^2).
    """
    aceleracion, info_deriv = derivar_con_consenso(
        velocidad_frenado,
        tiempo_frenado,
        tol_disp=tol_disp,
    )

    # La desaceleración máxima corresponde al valor más negativo de a(t) en magnitud
    desaceleracion_max = float(np.max(-aceleracion))
    limite_confort = 0.3 * g  # ~ 2.943 m/s^2
    supera_confort = desaceleracion_max > limite_confort

    resumen_fisico = {
        "desaceleracion_max": desaceleracion_max,
        "limite_confort": limite_confort,
        "supera_confort": supera_confort,
        "info_numerica": info_deriv,
    }
    return aceleracion, resumen_fisico