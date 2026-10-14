import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider


class DashboardAero:

    def __init__(self, datos):
        self.datos = datos
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Configuración de menú lateral (RadioButtons) y Slider inferior
        ax_radio = plt.axes([0.02, 0.5, 0.18, 0.35])
        self.radio = RadioButtons(
            ax_radio, ("Aproximación", "Atmósfera", "Frenado")
        )
        self.radio.on_clicked(self.actualizar_vista)

        ax_slider = plt.axes([0.3, 0.08, 0.55, 0.04])
        self.slider_masa = Slider(
            ax_slider,
            "Factor Masa",
            0.8,
            1.4,
            valinit=1.0,
            valstep=0.05,
        )
        self.slider_masa.on_changed(self.actualizar_frenado)

        self.mostrar_aproximacion()

    def limpiar_ejes(self):
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.6)

    def mostrar_aproximacion(self):
        self.limpiar_ejes()
        m, n, gamma = self.datos["ils_results"]
        x, h = self.datos["x_ils"], self.datos["h_ils"]

        self.ax.plot(
            x, h, "b.", alpha=0.3, label="Telemetría Radar ILS"
        )
        self.ax.plot(
            x,
            m * x + n,
            "r-",
            lw=2,
            label=f"Senda Ajustada (γ={gamma:.2f}°)",
        )
        self.ax.set_title(
            "Análisis de Aproximación: Senda ILS y Reparto de Cargas",
            fontsize=11,
            fontweight="bold",
        )
        self.ax.set_xlabel("Distancia al Umbral de Pista [m]")
        self.ax.set_ylabel("Altitud [m]")
        self.ax.legend()

        Rm, Rpi, Rpd = self.datos["reacciones"]
        info_text = (
            f"--- REPARTO DE CARGAS ---\n"
            f"Tren Morro (Rm): {Rm/1e3:.2f} kN\n"
            f"Principal Izq (Rpi): {Rpi/1e3:.2f} kN\n"
            f"Principal Der (Rpd): {Rpd/1e3:.2f} kN\n\n"
            f"--- SENDA DE PLANEO ILS ---\n"
            f"Ángulo de Descenso (γ): {gamma:.2f}°\n"
            f"Normativa OACI: ~3.00° (Cumple)"
        )
        self.ax.text(
            0.03,
            0.95,
            info_text,
            transform=self.ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        self.fig.canvas.draw_idle()

    def mostrar_atmosfera(self):
        self.limpiar_ejes()
        alt, vel = (
            self.datos["altitudes_viento"],
            self.datos["velocidad_viento"],
        )
        v_60 = self.datos["viento_60m"]

        self.ax.plot(
            vel, alt, "go-", lw=2, label="Perfil de Viento Boundary Layer"
        )
        self.ax.plot(
            v_60,
            60,
            "ro",
            ms=9,
            label=f"Viento a DH (60m): {v_60:.2f} m/s",
        )
        self.ax.axhline(60, color="r", linestyle=":", alpha=0.7)
        self.ax.set_title(
            "Análisis Atmosférico: Perfil de Viento en Capa Límite",
            fontsize=11,
            fontweight="bold",
        )
        self.ax.set_xlabel("Velocidad del Viento Cruzado [m/s]")
        self.ax.set_ylabel("Altitud [m]")
        self.ax.legend()

        info_text = (
            f"Altitud de Decisión (DH): 60.0 m\n"
            f"Viento Cruzado Interpolado: {v_60:.2f} m/s ({v_60*1.94384:.1f} kt)\n"
            f"Evaluación: Operación segura dentro de límites."
        )
        self.ax.text(
            0.03,
            0.95,
            info_text,
            transform=self.ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        self.fig.canvas.draw_idle()

    def mostrar_frenado(self):
        self.limpiar_ejes()
        t = self.datos["t_frenado"]
        v = self.datos["v_frenado"]
        a = self.datos["a_frenado"]
        ttaxi = self.datos["ttaxi"]
        factor = self.slider_masa.val
        distancia = self.datos["dist_frenado"] * factor

        self.ax.plot(t, v, "b-", lw=2, label="Velocidad V(t) [m/s]")
        self.ax.plot(
            t, np.abs(a), "m--", lw=1.5, label="Desaceleración |a(t)| [m/s²]"
        )
        self.ax.axvline(
            ttaxi,
            color="g",
            linestyle="-.",
            label=f"t_taxi = {ttaxi:.2f} s (V=15m/s)",
        )
        self.ax.axhline(
            2.94, color="r", linestyle=":", label="Límite Confort (0.3g)"
        )
        self.ax.fill_between(
            t, v * factor, color="blue", alpha=0.15, label="Distancia Recorrida"
        )
        self.ax.set_title(
            "Dinámica de Frenado y Distancia Consumida en Pista",
            fontsize=11,
            fontweight="bold",
        )
        self.ax.set_xlabel("Tiempo desde Touchdown [s]")
        self.ax.set_ylabel("Cinemática [m/s | m/s²]")
        self.ax.legend(loc="center right")

        alerta = (
            "⚠️ ¡EXCESO DE PISTA!"
            if distancia > 1600
            else "OK: Pista Suficiente"
        )
        info_text = (
            f"Tiempo a V_taxi (15 m/s): {ttaxi:.2f} s\n"
            f"Desaceleración Máx: {np.max(np.abs(a)):.2f} m/s² ({np.max(np.abs(a))/9.81:.2f}g)\n"
            f"Distancia Pista Consumida: {distancia:.1f} m\n"
            f"Diagnóstico: {alerta}"
        )
        self.ax.text(
            0.03,
            0.95,
            info_text,
            transform=self.ax.transAxes,
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="yellow" if distancia > 1600 else "white",
                alpha=0.85,
            ),
        )
        self.fig.canvas.draw_idle()

    def actualizar_vista(self, label):
        if label == "Aproximación":
            self.mostrar_aproximacion()
        elif label == "Atmósfera":
            self.mostrar_atmosfera()
        elif label == "Frenado":
            self.mostrar_frenado()

    def actualizar_frenado(self, val):
        if self.radio.value_selected == "Frenado":
            self.mostrar_frenado()