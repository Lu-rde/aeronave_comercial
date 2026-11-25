import numpy as np 
from math_utils import interpolar_con_consenso
from math_utils import derivar_con_consenso
from math_utils import integrar_con_consenso
from math_utils import resolver_raiz_con_consenso

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



def velocidad_frenado_analitica(t):
    """
    Perfil analítico de velocidad durante el frenado: V(t) = 70*exp(-0.04*t) - 1.2*t [m/s].
    """
    return 70.0 * np.exp(-0.04 * t) - 1.2 * t


def derivada_velocidad_frenado(t):
    """
    Derivada analítica de la velocidad (aceleración): dV/dt = -2.8*exp(-0.04*t) - 1.2 [m/s^2].
    """
    return -2.8 * np.exp(-0.04 * t) - 1.2


def calcular_tiempo_rodaje(v_taxi=15.0, t0=10.0, t_min=0.0, t_max=35.0):
    """
    Calcula el instante t_taxi [s] en el que la aeronave alcanza la velocidad segura
    de rodaje (V_taxi = 15 m/s) resolviendo f(t) = V(t) - V_taxi = 0.
    """
    f = lambda t: velocidad_frenado_analitica(t) - v_taxi
    df = lambda t: derivada_velocidad_frenado(t)

    t_taxi, info_raiz = resolver_raiz_con_consenso(
        f=f,
        df=df,
        x0=t0,
        a=t_min,
        b=t_max,
    )
    return t_taxi, info_raiz

# ==============================================================================
# [EXTRA] MARGEN OPERATIVO DE PISTA DESDE ALTURA DE DECISIÓN (DH)
# Definición: Combina la distancia horizontal recorrida en descenso desde la
# altura de decisión (DH = 60 m) con el ángulo de senda ILS más la distancia
# de frenado integrada S, comparándolo con una pista de referencia (2800 m)
# para comprobar cuánta longitud de pista sobra con seguridad.
# ==============================================================================
def evaluar_margen_pista_total(altitud_dh, gamma_deg, distancia_frenado_m, longitud_pista_ref=2800.0):
    """
    Integra las fases de aproximación y frenado: calcula la distancia horizontal
    desde la Altitud de Decisión (DH) hasta el umbral/toque aproximado más la
    distancia de roll-out S, y determina el margen libre sobre una pista estándar.
    """
    ang_rad = np.radians(abs(gamma_deg))
    dist_aprox_dh = float(altitud_dh / np.tan(ang_rad)) if ang_rad > 0 else 0.0
    dist_total_estimada = dist_aprox_dh + float(distancia_frenado_m)
    margen_m = float(longitud_pista_ref - dist_total_estimada)
    margen_pct = float((margen_m / longitud_pista_ref) * 100.0)

    return {
        "dist_aprox_dh": dist_aprox_dh,
        "dist_total_estimada": dist_total_estimada,
        "longitud_pista_ref": float(longitud_pista_ref),
        "margen_m": margen_m,
        "margen_pct": margen_pct,
    }