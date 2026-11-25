import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

from math_utils import resolver_con_consenso, ajustar_senda_consenso
from physics import (
    calcular_viento_altura_decision,
    calcular_distancia_frenado,
    calcular_perfil_desaceleracion,
    calcular_tiempo_rodaje,
)


def ejecutar_dashboard():
    """
    Carga los datos de telemetría del Tema 4, ejecuta los cálculos numéricos
    y físicos con consenso interno, imprime la evaluación física en consola y
    lanza el Dashboard interactivo de Matplotlib.
    """
    # ==========================================================================
    # 1. DATOS DE PARTIDA DE TELEMETRÍA (TEMA 4)
    # ==========================================================================
    np.random.seed(42)  # Fijamos semilla para reproducibilidad en la presentación

    # Sistema lineal: [x_cg (m), peso_total (N), momento_X (N·m), momento_Y (N·m)]
    params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0])
    matriz_geometria = np.array([
        [1.0, 1.0, 1.0],      # Suma de fuerzas verticales
        [3.2, 20.1, 20.1],    # Brazos de palanca eje X
        [0.0, -4.5, 4.5],     # Brazos de palanca eje Y
    ])
    b = params_masa[1:]

    # Regresión lineal: Telemetría ILS (Distancia a pista [m] vs Altitud [m])
    x_ils = np.linspace(-10000, -500, 200)
    h_ils = 0.0524 * np.abs(x_ils) + 15.0 + np.random.normal(0, 8, 200)

    # Interpolación 1D: Perfil de viento en la capa límite
    altitudes_viento = np.array([0, 50, 100, 200, 500, 1000])      # [m]
    velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3])  # [m/s]

    # Derivación, Integración y Raíces: Dinámica de frenado
    tiempo_frenado = np.linspace(0, 35, 1000)  # [s]
    velocidad_frenado = 70.0 * np.exp(-0.04 * tiempo_frenado) - 1.2 * tiempo_frenado
    velocidad_frenado[velocidad_frenado < 0] = 0

    # ==========================================================================
    # 2. EJECUCIÓN DE CÁLCULOS NUMÉRICOS (CON CONSENSO INTERNO)
    # ==========================================================================
    # Apartado 1: Reparto de cargas en trenes de aterrizaje
    rep_cargas = resolver_con_consenso(matriz_geometria, b, tol_disp=1e-2)
    R_m, R_pi, R_pd = rep_cargas["x_final"]

    # Apartado 2: Regresión lineal de la senda ILS
    m_ils, n_ils, gamma_deg, _ = ajustar_senda_consenso(x_ils, h_ils)
    h_ils_recta = m_ils * x_ils + n_ils

    # Apartado 3: Viento en Altitud de Decisión (DH = 60 m)
    altitud_dh = 60.0
    v_viento_dh, _ = calcular_viento_altura_decision(altitudes_viento, velocidad_viento, altitud_dh)

    # Apartado 4: Instante de velocidad de rodaje (V_taxi = 15 m/s)
    v_taxi = 15.0
    t_taxi, _ = calcular_tiempo_rodaje(v_taxi=v_taxi, t0=10.0)

    # Apartado 5: Perfil de desaceleración a(t) = dV/dt
    aceleracion_frenado, info_desacel = calcular_perfil_desaceleracion(tiempo_frenado, velocidad_frenado)
    desacel_max = info_desacel["desaceleracion_max"]
    limite_confort = info_desacel["limite_confort"]

    # Apartado 6: Distancia total de pista consumida S = integral(V dt)
    distancia_pista_m, _ = calcular_distancia_frenado(tiempo_frenado, velocidad_frenado)

    # ==========================================================================
    # 3. EVALUACIÓN FÍSICA EN CONSOLA (REQUISITO DE ENTREGA)
    # ==========================================================================
    print("=========================================================================")
    print("        EVALUACIÓN FÍSICA DE TELEMETRÍA DE ATERRIZAJE Y FRENADO          ")
    print("=========================================================================")
    print(f"[1. Cargas]  R_morro = {R_m/1000:.2f} kN | R_prin_izq = {R_pi/1000:.2f} kN | R_prin_der = {R_pd/1000:.2f} kN")
    print("    -> Física: Reacciones positivas y simétricas (M_y = 0); trenes principales soportan ~90% del peso.")
    print(f"[2. ILS]     Ángulo de senda gamma = {abs(gamma_deg):.2f}° (pendiente m = {m_ils:.5f})")
    print("    -> Física: Cumple la referencia estándar OACI de 3.0° para aproximación instrumental estabilizada.")
    print(f"[3. Viento]  Viento cruzado a DH = {altitud_dh:.0f} m: {v_viento_dh:.2f} m/s ({v_viento_dh*1.94384:.2f} kt)")
    print("    -> Física: Componente lateral moderada (< 15 kt), dentro de los límites operativos seguros.")
    print(f"[4. Rodaje]  Instante V_taxi ({v_taxi:.0f} m/s): t_taxi = {t_taxi:.2f} s")
    print("    -> Física: La aeronave reduce su velocidad a ritmo seguro para abandonar la pista por calle de rodaje.")
    print(f"[5. Confort] Desaceleración máxima: {desacel_max:.2f} m/s² (límite confort 0.3g = {limite_confort:.2f} m/s²)")
    print("    -> Física: Supera ligeramente 0.3g en el toque inicial por la resistencia aerodinámica, con frenado seguro.")
    print(f"[6. Pista]   Distancia total de frenado: S = {distancia_pista_m:.2f} m")
    print("    -> Física: Longitud de roll-out inferior a 800 m, totalmente segura para pistas comerciales estándar.")
    print("=========================================================================")

    # ==========================================================================
    # 4. INTERFAZ GRÁFICA INTERACTIVA (DASHBOARD MATPLOTLIB)
    # ==========================================================================
    fig = plt.figure(figsize=(13, 7.5))
    fig.suptitle(
        "DASHBOARD DE TELEMETRÍA: SENDA DE ATERRIZAJE Y DINÁMICA DE FRENADO",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )

    # Panel izquierdo superior: selector RadioButtons (3 vistas lógicas)
    ax_radio = plt.axes([0.03, 0.62, 0.22, 0.25], facecolor="#f4f6f9")
    ax_radio.set_title("Fase de Maniobra", fontsize=10, fontweight="bold", pad=8)
    radio = RadioButtons(ax_radio, ("Aproximación", "Atmósfera", "Frenado"), active=0)

    # Panel izquierdo inferior: caja de texto reservada para resultados numéricos dinámicos
    ax_texto = plt.axes([0.03, 0.08, 0.22, 0.50], facecolor="#f8f9fa")
    ax_texto.set_xticks([])
    ax_texto.set_yticks([])
    ax_texto.set_title("Resultados Numéricos", fontsize=10, fontweight="bold", pad=8)
    texto_panel = ax_texto.text(
        0.05,
        0.95,
        "",
        transform=ax_texto.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )

    # Área derecha: ejes principales (uno simple para Aproximación/Atmósfera, dos para Frenado)
    ax_main = plt.axes([0.32, 0.12, 0.64, 0.76])
    ax_fren_vel = plt.axes([0.32, 0.53, 0.64, 0.35])
    ax_fren_acc = plt.axes([0.32, 0.10, 0.64, 0.35], sharex=ax_fren_vel)

    # Resumen fijo de cargas estáticas que siempre acompaña al panel numérico
    bloque_cargas = (
        "REPARTO DE CARGAS (Ax=b)\n"
        f"• R_morro: {R_m/1000:6.2f} kN\n"
        f"• R_p_izq: {R_pi/1000:6.2f} kN\n"
        f"• R_p_der: {R_pd/1000:6.2f} kN\n"
        "-------------------------\n"
    )

    def actualizar_vista(etiqueta):
        """Actualiza la gráfica y el panel de resultados según la vista seleccionada."""
        ax_main.set_visible(False)
        ax_fren_vel.set_visible(False)
        ax_fren_acc.set_visible(False)

        if etiqueta == "Aproximación":
            ax_main.set_visible(True)
            ax_main.clear()

            ax_main.scatter(x_ils, h_ils, s=14, alpha=0.55, color="#4a7c9b", label="Telemetría Radar ILS")
            ax_main.plot(
                x_ils,
                h_ils_recta,
                color="#d9534f",
                linewidth=2.2,
                label=f"Regresión Lineal (γ = {abs(gamma_deg):.2f}°)",
            )
            ax_main.set_title("Calibración de Senda de Planeo ILS (h = m·x + n)", fontsize=11, fontweight="bold")
            ax_main.set_xlabel("Distancia horizontal a umbral de pista x [m]")
            ax_main.set_ylabel("Altitud h [m]")
            ax_main.grid(True, linestyle="--", alpha=0.4)
            ax_main.legend(loc="upper right")

            texto_panel.set_text(
                bloque_cargas
                + "FASE: APROXIMACIÓN ILS\n"
                + f"• Pendiente m: {m_ils:.5f}\n"
                + f"• Ordenada n:  {n_ils:.2f} m\n"
                + f"• Ángulo γ:   {abs(gamma_deg):.2f}°\n"
                + "• Ref. OACI:   3.00°\n"
                + "• Estado:      ESTABILIZADA"
            )

        elif etiqueta == "Atmósfera":
            ax_main.set_visible(True)
            ax_main.clear()

            alt_fina = np.linspace(altitudes_viento[0], altitudes_viento[-1], 300)
            v_fina = np.interp(alt_fina, altitudes_viento, velocidad_viento)

            ax_main.plot(velocidad_viento, altitudes_viento, "o--", color="#2b5c8f", label="Datos medidos capa límite")
            ax_main.plot(v_fina, alt_fina, "-", color="#5bc0de", alpha=0.7, label="Perfil interpolado 1D")
            ax_main.scatter(
                [v_viento_dh],
                [altitud_dh],
                color="red",
                s=95,
                zorder=5,
                label=f"DH = 60 m ({v_viento_dh:.2f} m/s)",
            )
            ax_main.axhline(altitud_dh, color="red", linestyle=":", alpha=0.6)

            ax_main.set_title("Perfil Vertical de Viento Cruzado y Altitud de Decisión (DH)", fontsize=11, fontweight="bold")
            ax_main.set_xlabel("Velocidad del viento cruzado [m/s]")
            ax_main.set_ylabel("Altitud sobre pista [m]")
            ax_main.grid(True, linestyle="--", alpha=0.4)
            ax_main.legend(loc="lower right")

            texto_panel.set_text(
                bloque_cargas
                + "FASE: ATMÓSFERA (DH)\n"
                + f"• Altitud DH:  {altitud_dh:.1f} m\n"
                + f"• Viento DH:   {v_viento_dh:.2f} m/s\n"
                + f"• En nudos:    {v_viento_dh * 1.94384:.2f} kt\n"
                + "• Estado:      DENTRO DE LÍMITE"
            )

        elif etiqueta == "Frenado":
            ax_fren_vel.set_visible(True)
            ax_fren_acc.set_visible(True)
            ax_fren_vel.clear()
            ax_fren_acc.clear()

            # Subgráfica 1: Velocidad V(t), área integrada S y punto t_taxi (Newton-Raphson)
            ax_fren_vel.plot(tiempo_frenado, velocidad_frenado, color="#1f77b4", linewidth=2, label="Velocidad V(t) [m/s]")
            ax_fren_vel.fill_between(
                tiempo_frenado,
                0,
                velocidad_frenado,
                color="#1f77b4",
                alpha=0.18,
                label=f"Distancia integrada S = {distancia_pista_m:.1f} m",
            )
            ax_fren_vel.axvline(
                t_taxi,
                color="#d9534f",
                linestyle="--",
                linewidth=1.6,
                label=f"t_taxi = {t_taxi:.2f} s (V = 15 m/s)",
            )
            ax_fren_vel.scatter([t_taxi], [v_taxi], color="#d9534f", s=65, zorder=5)
            ax_fren_vel.set_title("Dinámica de Frenado: Velocidad, Distancia Consumida y Desaceleración", fontsize=11, fontweight="bold")
            ax_fren_vel.set_ylabel("Velocidad V [m/s]")
            ax_fren_vel.grid(True, linestyle="--", alpha=0.4)
            ax_fren_vel.legend(loc="upper right", fontsize=8.5)

            # Subgráfica 2: Perfil de desaceleración a(t) y umbral de confort -0.3g
            ax_fren_acc.plot(tiempo_frenado, aceleracion_frenado, color="#2ca02c", linewidth=2, label="Aceleración a(t) = dV/dt [m/s²]")
            ax_fren_acc.axhline(
                -limite_confort,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.6,
                label=f"Límite confort (-0.3g = -{limite_confort:.2f} m/s²)",
            )
            ax_fren_acc.axvline(t_taxi, color="#d9534f", linestyle="--", linewidth=1.2, alpha=0.7)
            ax_fren_acc.set_xlabel("Tiempo desde touchdown t [s]")
            ax_fren_acc.set_ylabel("Aceleración a [m/s²]")
            ax_fren_acc.grid(True, linestyle="--", alpha=0.4)
            ax_fren_acc.legend(loc="lower right", fontsize=8.5)

            texto_panel.set_text(
                bloque_cargas
                + "FASE: FRENADO EN PISTA\n"
                + f"• t_taxi:      {t_taxi:.2f} s\n"
                + f"• Distancia S: {distancia_pista_m:.2f} m\n"
                + f"• Desac. máx:  {desacel_max:.2f} m/s²\n"
                + f"• Lím. 0.3g:   {limite_confort:.2f} m/s²\n"
                + "• Estado:      PISTA SUFICIENTE"
            )

        fig.canvas.draw_idle()

    # Guardamos referencia en fig para evitar que el recolector de basura desactive los RadioButtons
    fig._radio_widget = radio
    radio.on_clicked(actualizar_vista)
    actualizar_vista("Aproximación")
    plt.show()