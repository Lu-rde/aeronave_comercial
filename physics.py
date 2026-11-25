from math_utils import interpolar_con_consenso


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