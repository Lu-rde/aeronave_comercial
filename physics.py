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