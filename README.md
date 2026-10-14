# Tema 4: Senda de Aterrizaje y Dinámica de Frenado de una Aeronave Comercial

**Asignatura:** Programación / Cálculo I — Grado en Ingeniería Aeroespacial  
**Proyecto:** Panel de Control (Dashboard) Interactivo para Análisis de Telemetría Aeroespacial  

---

## 📌 Descripción del Proyecto

Este proyecto desarrolla una herramienta computacional orientada a la oficina técnica de ingeniería aeroespacial para la evaluación cinemática, aerodinámica y estructural durante las fases de **aproximación, touchdown y frenado (roll-out)** de una aeronave comercial. 

El sistema procesa datos telemétricos simulados de sensores de a bordo y estaciones terrestres mediante métodos numéricos vectorizados de alto rendimiento en `numpy`, integrando la visualización interactiva en un **Dashboard de ventana única** programado con `matplotlib.widgets`.

---

## 📁 Estructura del Repositorio

El proyecto cumple de forma estricta con la arquitectura modular requerida por la guía docente:

```plaintext
PROYECTO_FINAL_CALCULO/
├── main.py                # Punto de entrada; carga la telemetría, ejecuta el reporte en consola y lanza la GUI.
├── calculos_aero.py       # Módulo puramente matemático con las funciones numéricas vectorizadas.
├── dashboard_gui.py       # Módulo de la Interfaz Gráfica de Usuario (GUI) interactiva con matplotlib.widgets.
└── README.md              # Documentación técnica del proyecto.