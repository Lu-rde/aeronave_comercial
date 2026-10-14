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
# 1. CARGA DE DATOS DE TELEMETRÍA (TEMA 4)
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
# 2. EJECUCIÓN DE CÁLCULOS NUMÉRICOS Y REPORTE EN CONSOLA
# ==============================================================================
if __name__ == "__main__":
    # Apartado 1: Reparto de cargas
    Rm, Rpi, Rpd = resolver_reparto_cargas(params_masa, matriz_geometria)
    print("=" * 65)
    print("EVALUACIÓN DE TELEMETRÍA DE ATERRIZAJE (TEMA 4)")
    print("=" * 65)
    print(
        f"[Apartado 1] Reacciones: Morro = {Rm/1e3:.2f} kN | Izq ="
        f" {Rpi/1e3:.2f} kN | Der = {Rpd/1e3:.2f} kN"
    )
    print(
        "-> Interpretación física: Reparto simétrico en el tren principal"
        " (Rpi = Rpd) debido a momento en Y nulo (0.0 Nm)."
    )

    # Apartado 2: Senda ILS
    m, n, gamma = calcular_senda_ils(x_ils, h_ils)
    print(
        f"[Apartado 2] Regresión ILS: m = {m:.5f} | n = {n:.2f} m | Ángulo"
        f" γ = {gamma:.2f}°"
    )
    print(
        "-> Interpretación física: Ángulo conforme con la normativa OACI"
        " (~3.00°)."
    )

    # Apartado 3: Viento cruzado
    v_60 = calcular_viento_decision(altitudes_viento, velocidad_viento, 60.0)
    print(
        f"[Apartado 3] Viento a Altitud de Decisión (60m): {v_60:.2f} m/s"
        f" ({v_60*1.94384:.1f} kt)"
    )

    # Apartado 4: Tiempo V_taxi
    ttaxi = calcular_tiempo_taxi(t0=10.0, v_target=15.0)
    print(
        f"[Apartado 4] Tiempo hasta velocidad de rodaje (15 m/s):"
        f" {ttaxi:.2f} s (Newton-Raphson)"
    )

    # Apartado 5: Desaceleración
    a_frenado = calcular_desaceleracion(tiempo_frenado, velocidad_frenado)
    a_max = np.max(np.abs(a_frenado))
    print(
        f"[Apartado 5] Desaceleración Máxima: {a_max:.2f} m/s² ({a_max/9.81:.2f}g)"
    )
    print(
        f"-> Interpretación física: {'Cumple' if a_max <= 2.94 else 'Supera'} el"
        " límite de confort de pasajeros (0.3g ≈ 2.94 m/s²)."
    )

    # Apartado 6: Distancia de frenado
    dist_frenado = calcular_distancia_frenado(
        tiempo_frenado, velocidad_frenado
    )
    print(
        f"[Apartado 6] Distancia total consumida en pista: {dist_frenado:.2f} m"
    )
    print("=" * 65 + "\nLanzando Dashboard interactivo...")

    # Paquete de datos para la GUI
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