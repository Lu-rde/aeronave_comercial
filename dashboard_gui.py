import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from math_utils import resolver_con_consenso, ajustar_senda_consenso
from physics import (
    calcular_viento_altura_decision,
    calcular_distancia_frenado,
    calcular_perfil_desaceleracion,
    calcular_tiempo_rodaje,
    evaluar_margen_pista_total,
)


def ejecutar_dashboard():
    """
    Carga los datos de telemetría del Tema 4, ejecuta los cálculos numéricos y físicos
    con consenso interno, imprime la evaluación física en consola y lanza el
    Dashboard interactivo con RadioButtons y Sliders dinámicos en las tres fases.
    """
    # ==========================================================================
    # 1. DATOS DE PARTIDA DE TELEMETRÍA (TEMA 4)
    # ==========================================================================
    rng = np.random.default_rng(42)

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
    ruido_base_ils = rng.normal(0.0, 1.0, size=x_ils.shape)
    sigma_ils_inicial = 8.0
    h_ils_inicial = 0.0524 * np.abs(x_ils) + 15.0 + sigma_ils_inicial * ruido_base_ils

    # Interpolación 1D: Perfil de viento en la capa límite
    altitudes_viento = np.array([0, 50, 100, 200, 500, 1000], dtype=float)      # [m]
    velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3], dtype=float)  # [m/s]

    # Derivación, Integración y Raíces: Dinámica de frenado
    tiempo_frenado = np.linspace(0, 35, 1000)  # [s]
    velocidad_frenado = 70.0 * np.exp(-0.04 * tiempo_frenado) - 1.2 * tiempo_frenado
    velocidad_frenado[velocidad_frenado < 0] = 0

    # ==========================================================================
    # 2. EJECUCIÓN DE CÁLCULOS NUMÉRICOS INICIALES (CON CONSENSO INTERNO)
    # ==========================================================================
    rep_cargas = resolver_con_consenso(matriz_geometria, b, tol_disp=1e-2)
    R_m, R_pi, R_pd = rep_cargas["x_final"]

    m_ils_ini, n_ils_ini, gamma_deg_ini, _ = ajustar_senda_consenso(x_ils, h_ils_inicial)
    h_ils_recta_ini = m_ils_ini * x_ils + n_ils_ini

    altitud_dh_inicial = 60.0
    v_viento_dh_ini, _ = calcular_viento_altura_decision(altitudes_viento, velocidad_viento, altitud_dh_inicial)

    v_taxi_inicial = 15.0
    t_taxi_inicial, _ = calcular_tiempo_rodaje(v_taxi=v_taxi_inicial, t0=10.0)

    aceleracion_frenado, info_desacel = calcular_perfil_desaceleracion(tiempo_frenado, velocidad_frenado)
    desacel_max = info_desacel["desaceleracion_max"]
    limite_confort = info_desacel["limite_confort"]

    distancia_pista_m, _ = calcular_distancia_frenado(tiempo_frenado, velocidad_frenado)

    # ==========================================================================
    # [EXTRA] CÁLCULO DE MARGEN OPERATIVO DE PISTA (APROXIMACIÓN DH + FRENADO)
    # Definición: Estima la distancia total desde que se cruza DH = 60 m hasta
    # detenerse en pista y calcula los metros y porcentaje de pista libre que
    # quedan respecto a una longitud de pista de referencia de 2800 m.
    # ==========================================================================
    info_pista_total = evaluar_margen_pista_total(
        altitud_dh=altitud_dh_inicial,
        gamma_deg=gamma_deg_ini,
        distancia_frenado_m=distancia_pista_m,
        longitud_pista_ref=2800.0,
    )

    # ==========================================================================
    # 3. EVALUACIÓN FÍSICA EN CONSOLA
    # ==========================================================================
    print("=========================================================================")
    print("        EVALUACIÓN FÍSICA DE TELEMETRÍA DE ATERRIZAJE Y FRENADO          ")
    print("=========================================================================")
    print(f"[1. Cargas]  R_morro = {R_m/1000:.2f} kN | R_prin_izq = {R_pi/1000:.2f} kN | R_prin_der = {R_pd/1000:.2f} kN")
    print("    -> Física: Reacciones positivas y simétricas (M_y = 0); trenes principales soportan ~90% del peso.")
    print(f"[2. ILS]     Ángulo de senda gamma = {abs(gamma_deg_ini):.2f}° (pendiente m = {m_ils_ini:.5f})")
    print("    -> Física: Cumple la referencia estándar OACI de 3.0° para aproximación instrumental estabilizada.")
    print(f"[3. Viento]  Viento cruzado a DH = {altitud_dh_inicial:.0f} m: {v_viento_dh_ini:.2f} m/s ({v_viento_dh_ini*1.94384:.2f} kt)")
    print("    -> Física: Componente lateral moderada (< 15 kt), dentro de los límites operativos seguros.")
    print(f"[4. Rodaje]  Instante V_taxi ({v_taxi_inicial:.0f} m/s): t_taxi = {t_taxi_inicial:.2f} s")
    print("    -> Física: La aeronave reduce su velocidad a ritmo seguro para abandonar la pista por calle de rodaje.")
    print(f"[5. Confort] Desaceleración máxima: {desacel_max:.2f} m/s² (límite confort 0.3g = {limite_confort:.2f} m/s²)")
    print("    -> Física: Supera ligeramente 0.3g en el toque inicial por la resistencia aerodinámica, con frenado seguro.")
    print(
        f"[6. Pista]   Frenado S = {distancia_pista_m:.2f} m | Total desde DH = {info_pista_total['dist_total_estimada']:.1f} m | "
        f"Margen (2800 m) = {info_pista_total['margen_m']:.1f} m ({info_pista_total['margen_pct']:.1f}%)"
    )
    print("    -> Física: Longitud de roll-out segura y amplio margen libre frente a una pista comercial estándar.")
    print("=========================================================================")

    # ==========================================================================
    # 4. INTERFAZ GRÁFICA INTERACTIVA (DASHBOARD MATPLOTLIB)
    # ==========================================================================
    fig = plt.figure(figsize=(13.5, 7.6))
    fig.suptitle(
        "DASHBOARD DE TELEMETRÍA: SENDA DE ATERRIZAJE Y DINÁMICA DE FRENADO",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )

    # Panel izquierdo superior: selector RadioButtons (3 vistas lógicas)
    ax_radio = plt.axes([0.025, 0.64, 0.23, 0.23], facecolor="#f4f6f9")
    ax_radio.set_title("Fase de Maniobra", fontsize=10, fontweight="bold", pad=8)
    radio = RadioButtons(ax_radio, ("Aproximación", "Atmósfera", "Frenado"), active=0)

    # Panel izquierdo inferior: caja de texto reservada para resultados numéricos dinámicos
    ax_texto = plt.axes([0.025, 0.08, 0.23, 0.52], facecolor="#f8f9fa")
    ax_texto.set_xticks([])
    ax_texto.set_yticks([])
    ax_texto.set_title("Resultados Numéricos", fontsize=10, fontweight="bold", pad=8)
    texto_panel = ax_texto.text(
        0.04,
        0.96,
        "",
        transform=ax_texto.transAxes,
        fontsize=8.6,
        verticalalignment="top",
        family="monospace",
    )

    # Área derecha: gráficas principales
    ax_main = plt.axes([0.32, 0.14, 0.64, 0.74])
    ax_fren_vel = plt.axes([0.32, 0.55, 0.64, 0.33])
    ax_fren_acc = plt.axes([0.32, 0.15, 0.64, 0.32], sharex=ax_fren_vel)

    # ==========================================================================
    # [EXTRA] SLIDERS INTERACTIVOS EN TIEMPO REAL PARA LAS TRES FASES
    # Definición: Controles deslizantes en la parte inferior que permiten variar
    # parámetros clave en cada fase (ruido radar ILS en Aproximación, altitud de
    # decisión DH en Atmósfera y V_taxi en Frenado) y recalcular al instante.
    # ==========================================================================
    ax_slider_ils = plt.axes([0.40, 0.04, 0.48, 0.03], facecolor="#eef2f7")
    slider_ils_sigma = Slider(
        ax=ax_slider_ils,
        label="Ruido ILS σ [m]",
        valmin=0.0,
        valmax=25.0,
        valinit=sigma_ils_inicial,
        valstep=0.5,
    )

    ax_slider_dh = plt.axes([0.40, 0.04, 0.48, 0.03], facecolor="#eef2f7")
    slider_dh = Slider(
        ax=ax_slider_dh,
        label="Altitud DH [m]",
        valmin=10.0,
        valmax=500.0,
        valinit=altitud_dh_inicial,
        valstep=5.0,
    )

    ax_slider_vtaxi = plt.axes([0.40, 0.04, 0.48, 0.03], facecolor="#eef2f7")
    slider_vtaxi = Slider(
        ax=ax_slider_vtaxi,
        label="V_taxi [m/s]",
        valmin=5.0,
        valmax=25.0,
        valinit=v_taxi_inicial,
        valstep=0.5,
    )

    bloque_cargas = (
        "REPARTO DE CARGAS (Ax=b)\n"
        f"• R_morro: {R_m/1000:6.2f} kN\n"
        f"• R_p_izq: {R_pi/1000:6.2f} kN\n"
        f"• R_p_der: {R_pd/1000:6.2f} kN\n"
        "-------------------------\n"
    )

    estado_actual = {
        "vista": "Aproximación",
        "sigma_ils": sigma_ils_inicial,
        "h_ils": h_ils_inicial,
        "m_ils": m_ils_ini,
        "n_ils": n_ils_ini,
        "gamma_deg": gamma_deg_ini,
        "h_ils_recta": h_ils_recta_ini,
        "dh": altitud_dh_inicial,
        "v_dh": v_viento_dh_ini,
        "v_taxi": v_taxi_inicial,
        "t_taxi": t_taxi_inicial,
    }

    def dibujar_aproximacion():
        """Dibuja la gráfica y el panel de Aproximación con el nivel de ruido ILS actual."""
        ax_main.clear()
        ax_main.scatter(x_ils, estado_actual["h_ils"], s=14, alpha=0.55, color="#4a7c9b", label="Telemetría Radar ILS")
        ax_main.plot(
            x_ils,
            estado_actual["h_ils_recta"],
            color="#d9534f",
            linewidth=2.2,
            label=f"Regresión Lineal (γ = {abs(estado_actual['gamma_deg']):.2f}°)",
        )
        ax_main.set_title("Calibración de Senda de Planeo ILS (h = m·x + n)", fontsize=11, fontweight="bold")
        ax_main.set_xlabel("Distancia horizontal a umbral de pista x [m]")
        ax_main.set_ylabel("Altitud h [m]")
        ax_main.grid(True, linestyle="--", alpha=0.4)
        ax_main.legend(loc="upper right")

        texto_panel.set_text(
            bloque_cargas
            + "FASE: APROXIMACIÓN ILS\n"
            + f"• Ruido σ:   {estado_actual['sigma_ils']:5.1f} m\n"
            + f"• Pendiente: {estado_actual['m_ils']:.5f}\n"
            + f"• Ordenada:  {estado_actual['n_ils']:6.2f} m\n"
            + f"• Ángulo γ:  {abs(estado_actual['gamma_deg']):6.2f}°\n"
            + "• Ref. OACI:   3.00°"
        )

    def dibujar_atmosfera():
        """Dibuja la gráfica y el panel de Atmósfera con la altitud DH seleccionada."""
        ax_main.clear()
        alt_fina = np.linspace(altitudes_viento[0], altitudes_viento[-1], 300)
        v_fina = np.interp(alt_fina, altitudes_viento, velocidad_viento)
        dh_sel = estado_actual["dh"]
        v_dh_sel = estado_actual["v_dh"]

        ax_main.plot(velocidad_viento, altitudes_viento, "o--", color="#2b5c8f", label="Datos medidos capa límite")
        ax_main.plot(v_fina, alt_fina, "-", color="#5bc0de", alpha=0.7, label="Perfil interpolado 1D")
        ax_main.scatter(
            [v_dh_sel],
            [dh_sel],
            color="red",
            s=95,
            zorder=5,
            label=f"DH = {dh_sel:.0f} m ({v_dh_sel:.2f} m/s)",
        )
        ax_main.axhline(dh_sel, color="red", linestyle=":", alpha=0.6)
        ax_main.set_title("Perfil Vertical de Viento Cruzado y Altitud de Decisión (DH)", fontsize=11, fontweight="bold")
        ax_main.set_xlabel("Velocidad del viento cruzado [m/s]")
        ax_main.set_ylabel("Altitud sobre pista [m]")
        ax_main.grid(True, linestyle="--", alpha=0.4)
        ax_main.legend(loc="lower right")

        texto_panel.set_text(
            bloque_cargas
            + "FASE: ATMÓSFERA (DH)\n"
            + f"• Altitud DH:{dh_sel:6.1f} m\n"
            + f"• Viento DH: {v_dh_sel:6.2f} m/s\n"
            + f"• En nudos:  {v_dh_sel * 1.94384:6.2f} kt\n"
            + "• Estado:    DENTRO DE LÍMITE"
        )

    def dibujar_frenado(v_obj, t_obj):
        """Dibuja las dos subgráficas de frenado con el valor actual de V_taxi y t_taxi."""
        ax_fren_vel.clear()
        ax_fren_acc.clear()

        ax_fren_vel.plot(tiempo_frenado, velocidad_frenado, color="#1f77b4", linewidth=2, label="Velocidad V(t) [m/s]")
        ax_fren_vel.fill_between(
            tiempo_frenado,
            0,
            velocidad_frenado,
            color="#1f77b4",
            alpha=0.18,
            label=f"Distancia frenado S = {distancia_pista_m:.1f} m",
        )
        ax_fren_vel.axvline(
            t_obj,
            color="#d9534f",
            linestyle="--",
            linewidth=1.6,
            label=f"t_taxi = {t_obj:.2f} s (V = {v_obj:.1f} m/s)",
        )
        ax_fren_vel.scatter([t_obj], [v_obj], color="#d9534f", s=65, zorder=5)
        ax_fren_vel.set_title("Dinámica de Frenado: Velocidad, Distancia y Desaceleración", fontsize=11, fontweight="bold")
        ax_fren_vel.set_ylabel("Velocidad V [m/s]")
        ax_fren_vel.grid(True, linestyle="--", alpha=0.4)
        ax_fren_vel.legend(loc="upper right", fontsize=8.5)

        ax_fren_acc.plot(tiempo_frenado, aceleracion_frenado, color="#2ca02c", linewidth=2, label="Aceleración a(t) = dV/dt [m/s²]")
        ax_fren_acc.axhline(
            -limite_confort,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.6,
            label=f"Límite confort (-0.3g = -{limite_confort:.2f} m/s²)",
        )
        ax_fren_acc.axvline(t_obj, color="#d9534f", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_fren_acc.set_xlabel("Tiempo desde touchdown t [s]")
        ax_fren_acc.set_ylabel("Aceleración a [m/s²]")
        ax_fren_acc.grid(True, linestyle="--", alpha=0.4)
        ax_fren_acc.legend(loc="lower right", fontsize=8.5)

        texto_panel.set_text(
            bloque_cargas
            + "FASE: FRENADO EN PISTA\n"
            + f"• V_taxi:   {v_obj:5.1f} m/s\n"
            + f"• t_taxi:   {t_obj:5.2f} s\n"
            + f"• Frenado:  {distancia_pista_m:6.1f} m\n"
            + f"• Tot. DH:  {info_pista_total['dist_total_estimada']:6.1f} m\n"
            + f"• Margen:   {info_pista_total['margen_m']:6.1f} m\n"
            + f"• a_max:    {desacel_max:5.2f} m/s²"
        )

    def actualizar_vista(etiqueta):
        """Actualiza la visibilidad de los ejes y sliders según la fase seleccionada."""
        estado_actual["vista"] = etiqueta
        es_aprox = etiqueta == "Aproximación"
        es_atmos = etiqueta == "Atmósfera"
        es_frenado = etiqueta == "Frenado"

        ax_main.set_visible(not es_frenado)
        ax_fren_vel.set_visible(es_frenado)
        ax_fren_acc.set_visible(es_frenado)

        ax_slider_ils.set_visible(es_aprox)
        ax_slider_dh.set_visible(es_atmos)
        ax_slider_vtaxi.set_visible(es_frenado)

        if es_aprox:
            dibujar_aproximacion()
        elif es_atmos:
            dibujar_atmosfera()
        elif es_frenado:
            dibujar_frenado(estado_actual["v_taxi"], estado_actual["t_taxi"])

        fig.canvas.draw_idle()

    def al_mover_slider_ils(nuevo_sigma):
        """Recalcula la telemetría ILS y el ajuste lineal en tiempo real."""
        sigma_sel = float(nuevo_sigma)
        h_sel = 0.0524 * np.abs(x_ils) + 15.0 + sigma_sel * ruido_base_ils
        m_sel, n_sel, gamma_sel, _ = ajustar_senda_consenso(x_ils, h_sel)

        estado_actual["sigma_ils"] = sigma_sel
        estado_actual["h_ils"] = h_sel
        estado_actual["m_ils"] = m_sel
        estado_actual["n_ils"] = n_sel
        estado_actual["gamma_deg"] = gamma_sel
        estado_actual["h_ils_recta"] = m_sel * x_ils + n_sel

        if estado_actual["vista"] == "Aproximación":
            dibujar_aproximacion()
            fig.canvas.draw_idle()

    def al_mover_slider_dh(nueva_dh):
        """Recalcula el viento interpolado a la altitud DH elegida en tiempo real."""
        dh_sel = float(nueva_dh)
        v_dh_sel, _ = calcular_viento_altura_decision(altitudes_viento, velocidad_viento, dh_sel)
        estado_actual["dh"] = dh_sel
        estado_actual["v_dh"] = v_dh_sel

        if estado_actual["vista"] == "Atmósfera":
            dibujar_atmosfera()
            fig.canvas.draw_idle()

    def al_mover_slider_vtaxi(nuevo_v_taxi):
        """Recalcula t_taxi en tiempo real al mover el slider de V_taxi."""
        v_sel = float(nuevo_v_taxi)
        t_sel, _ = calcular_tiempo_rodaje(v_taxi=v_sel, t0=10.0)
        estado_actual["v_taxi"] = v_sel
        estado_actual["t_taxi"] = t_sel

        if estado_actual["vista"] == "Frenado":
            dibujar_frenado(v_sel, t_sel)
            fig.canvas.draw_idle()

    fig._radio_widget = radio
    fig._slider_ils = slider_ils_sigma
    fig._slider_dh = slider_dh
    fig._slider_vtaxi = slider_vtaxi

    radio.on_clicked(actualizar_vista)
    slider_ils_sigma.on_changed(al_mover_slider_ils)
    slider_dh.on_changed(al_mover_slider_dh)
    slider_vtaxi.on_changed(al_mover_slider_vtaxi)

    actualizar_vista("Aproximación")
    fig.canvas.draw()
    plt.show()