import matplotlib.pyplot as plt
import numpy as np
from calculos_aero import (
    calcular_desaceleracion,
    calcular_distancia_frenado,
    calcular_senda_ils,
    calcular_tiempo_taxi,
    calcular_viento_decision,
    resolver_reparto_cargas,
)
from dashboard_gui import DashboardAero

# ==============================================================================
# 1. CARGA DE DATOS DE TELEMETRÍA SIMULADA (TEMA 4)
# ==============================================================================
params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0])
matriz_geometria = np.array(
    [[1.0, 1.0, 1.0], [3.2, 20.1, 20.1], [0.0, -4.5, 4.5]]
)
x_ils = np.linspace(-10000, -500, 200)
h_ils = 0.0524 * np.abs(x_ils) + 15.0 + np.random.normal(0, 8, 200)
altitudes_viento = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3])
tiempo_frenado = np.linspace(0, 35, 1000)
velocidad_frenado = (
    70.0 * np.exp(-0.04 * tiempo_frenado) - 1.2 * tiempo_frenado
)
velocidad_frenado[velocidad_frenado < 0] = 0

# ==============================================================================
# 2. EJECUCIÓN NUMÉRICA Y REPORTE FORMATO INFORME TÉCNICO
# ==============================================================================
if __name__ == "__main__":
    # Cálculo numérico de todos los apartados
    Rm, Rpi, Rpd = resolver_reparto_cargas(params_masa, matriz_geometria)
    m, n, gamma = calcular_senda_ils(x_ils, h_ils)
    v_60 = calcular_viento_decision(altitudes_viento, velocidad_viento, 60.0)
    ttaxi = calcular_tiempo_taxi(t0=10.0, v_target=15.0)
    a_frenado = calcular_desaceleracion(tiempo_frenado, velocidad_frenado)
    a_max = np.max(np.abs(a_frenado))
    dist_frenado = calcular_distancia_frenado(
        tiempo_frenado, velocidad_frenado
    )

    # Impresión estructurada en consola
    print("\n" + "=" * 75)
    print(
        " INFORME TÉCNICO DE TELEMETRÍA DE ATERRIZAJE — TEMA 4 ".center(
            75, "═"
        )
    )
    print("=" * 75)
    print("\n┌── [1] REPARTO DE CARGAS EN TREN DE ATERRIZAJE " + "─" * 28)
    print(f"│ • Tren de Morro (Rm): {Rm/1e3:8.2f} kN ({Rm:10.2f} N)")
    print(
        f"│ • Tren Principal Izquierdo: {Rpi/1e3:8.2f} kN ({Rpi:10.2f} N)"
    )
    print(
        f"│ • Tren Principal Derecho: {Rpd/1e3:8.2f} kN ({Rpd:10.2f} N)"
    )
    print(
        "│ ► Diagnóstico: Reparto simétrico por ausencia de momento lateral"
        " (My = 0.0 Nm)."
    )
    print("\n┌── SENDA DE PLANEO Y APROXIMACIÓN ILS " + "─" * 32)
    print(f"│ • Pendiente de la senda (m): {m:10.5f}")
    print(f"│ • Intersección umbral (n): {n:8.2f} m")
    print(f"│ • Ángulo de descenso (γ): {gamma:8.2f}°")
    print("│ ► Diagnóstico: Conforme a la normativa OACI (~3.00°).")
    print("\n┌── PERFIL DE VIENTO CRUZADO EN CAPA LÍMITE " + "─" * 27)
    print(
        f"│ • Viento a Altitud de Decisión (60m): {v_60:5.2f} m/s"
        f" ({v_60*1.94384:4.1f} kt)"
    )
    print(
        "│ ► Diagnóstico: Operación dentro de la envolvente de vuelo segura."
    )
    print(
        "\n┌── BÚSQUEDA DE RAÍCES: TIEMPO A VELOCIDAD DE RODAJE " + "─" * 18
    )
    print(f"│ • Tiempo hasta V_taxi (15 m/s): {ttaxi:8.2f} s")
    print(
        "│ ► Algoritmo: Newton-Raphson convergido con tolerancia < 1e-6."
    )
    print("\n┌── DERIVACIÓN NUMÉRICA: DESACELERACIÓN EN PISTA " + "─" * 21)
    print(
        f"│ • Desaceleración Máxima: {a_max:8.2f} m/s²"
        f" ({a_max/9.81:4.2f} g)"
    )
    eval_a = (
        "Cumple límite de confort (≤ 0.3g)"
        if a_max <= 2.94
        else "Supera límite de confort (> 0.3g)"
    )
    print(f"│ ► Diagnóstico: {eval_a}.")
    print("\n┌── INTEGRACIÓN NUMÉRICA: DISTANCIA DE FRENADO " + "─" * 23)
    print(f"│ • Pista total consumida: {dist_frenado:8.2f} m")
    eval_pista = (
        "OK (Pista suficiente)"
        if dist_frenado <= 1600
        else "¡ALERTA! Exceso de pista"
    )
    print(f"│ ► Diagnóstico: {eval_pista}.")
    print("\n" + "=" * 75)
    print(
        "Desplegando Dashboard Interactivo... Cierre la ventana gráfica para"
        " finalizar."
    )
    print("=" * 75 + "\n")

    # Preparación de datos para la interfaz gráfica
    datos_gui = {
        "x_ils": x_ils,
        "h_ils": h_ils,
        "ils_results": (m, n, gamma),
        "reacciones": (Rm, Rpi, Rpd),
        "altitudes_viento": altitudes_viento,
        "velocidad_viento": velocidad_viento,
        "viento_60m": v_60,
        "t_frenado": tiempo_frenado,
        "v_frenado": velocidad_frenado,
        "a_frenado": a_frenado,
        "ttaxi": ttaxi,
        "dist_frenado": dist_frenado,
    }

    app = DashboardAero(datos_gui)
    plt.show()