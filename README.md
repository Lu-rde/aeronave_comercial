# Tema 4: Senda de Aterrizaje y Dinámica de Frenado de una Aeronave Comercial

**Asignatura:** Programación / Cálculo I — Grado en Ingeniería Aeroespacial  
**Proyecto:** Panel de Control (Dashboard) Interactivo para Análisis de Telemetría Aeroespacial  

# Análisis Numérico y Telemetría de Aterrizaje en Aeronave Comercial (Tema 4)

Aplicación modular en Python para el análisis numérico y físico de la fase de aproximación, transición atmosférica y carrera de frenado (*roll-out*) de una aeronave comercial, acompañada de una consola de evaluación física y un *dashboard* interactivo en `matplotlib`.

---

## Tabla de contenidos

1. [Arquitectura del proyecto](#arquitectura-del-proyecto)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Ejecución y flujo de trabajo con Git](#ejecución-y-flujo-de-trabajo-con-git)
4. [Bloques numéricos y consenso de métodos](#bloques-numéricos-y-consenso-de-métodos)
5. [Modelado físico y mejoras extra (`[EXTRA]`)](#modelado-físico-y-mejoras-extra-extra)
6. [Dashboard interactivo](#dashboard-interactivo)
7. [Criterios técnicos y diseño vectorizado](#criterios-técnicos-y-diseño-vectorizado)

---

## Arquitectura del proyecto

El código está organizado en cuatro módulos con separación de responsabilidades:

```text
aeronave_comercial/
├── main.py            # Punto de entrada único; invoca ejecutar_dashboard()
├── dashboard_gui.py   # Carga de datos, ejecución de cálculos, salida física y GUI interactiva
├── physics.py         # Modelos físicos aeronáuticos y funciones extra de análisis operativo
└── math_utils.py      # Algoritmos numéricos vectorizados y mecanismos de consenso
```

| Módulo | Responsabilidad principal |
| :--- | :--- |
| `math_utils.py` | Implementación de métodos numéricos de sistemas lineales, regresión, interpolación, integración, derivación y búsqueda de raíces, junto con funciones de consenso y control de dispersión. |
| `physics.py` | Traducción de los resultados numéricos a magnitudes aeronáuticas (viento en altura de decisión, distancia de frenado, desaceleración, tiempo de rodaje y margen de pista). |
| `dashboard_gui.py` | Orquestación de los datos de telemetría, impresión del resumen físico en consola y renderizado de la ventana interactiva con `RadioButtons` y `Slider`. |
| `main.py` | Lanzamiento limpio de la aplicación mediante `ejecutar_dashboard()`. |

---

## Requisitos e instalación

El proyecto requiere **Python 3.10+** y las siguientes librerías científicas:

* `numpy`
* `scipy`
* `matplotlib`

Instalación de dependencias:

```powershell
pip install numpy scipy matplotlib
```

---

## Ejecución y flujo de trabajo con Git

### Ejecutar la aplicación

Desde la carpeta raíz del proyecto en PowerShell:

```powershell
python main.py
```

### Trabajar en una rama distinta de la principal

```powershell
git checkout -b feature/dashboard-consenso
git add .
git commit -m "Implementación completa de métodos numéricos, consenso y dashboard"
git push -u origin feature/dashboard-consenso
```

---

## Bloques numéricos y consenso de métodos

Cada bloque numérico evalúa múltiples implementaciones, mide la dispersión entre ellas y aplica una regla de consenso interna: si la dispersión está dentro de la tolerancia configurada, se promedian las soluciones válidas; si supera el umbral, se prioriza el método más robusto o adecuado al modelo.

1. **Sistemas de ecuaciones lineales (`resolver_con_consenso`)**
   * **Métodos:** eliminación gaussiana con pivoteo parcial (`my_Gauss`), solver directo de NumPy (`Gauss_numpy`), factorización LU con SciPy (`LU`), y métodos iterativos (`my_Jacobi` y `my_Gauss_Seidel`).
   * **Contexto:** equilibrio estático de fuerzas verticales y momentos para determinar las reacciones en los trenes de aterrizaje (morro, principal izquierdo y principal derecho).

2. **Regresión lineal (`ajustar_senda_consenso`)**
   * **Métodos:** ecuaciones normales matriciales por mínimos cuadrados (`my_regression`) y ajuste polinómico de grado 1 (`regression_numpy`).
   * **Contexto:** calibración de la recta de senda de planeo ILS $h = m \cdot x + n$ a partir de datos radar con ruido y cálculo del ángulo de descenso $\gamma = \arctan(m)$ frente a la referencia OACI de $\approx 3^\circ$.

3. **Interpolación 1D (`interpolar_con_consenso`)**
   * **Métodos:** interpolación lineal por tramos (`interp_numpy`), interpolación polinómica global mediante matriz de Vandermonde (`interp_vandermonde`) y base de Lagrange vectorizada (`interp_lagrange`).
   * **Contexto:** estimación de la velocidad del viento en la **Altura de Decisión** ($\text{DH} = 60\text{ m}$) a partir del perfil discreto de la capa límite atmosférica.

4. **Integración numérica 1D (`integrar_con_consenso`)**
   * **Métodos:** regla del trapecio compuesta (`my_trapecio`), regla de Simpson compuesta (`my_simpson`) e integración con NumPy (`integracion_numpy`).
   * **Contexto:** cálculo de la distancia de pista consumida durante la carrera de frenado $S = \int V(t)\,dt$.

5. **Derivación numérica 1D (`derivar_con_consenso`)**
   * **Métodos:** diferencias finitas centrales de 3 puntos con esquemas de 3 puntos en extremos (`my_derivada_central`), esquemas de 3 puntos adelantados/retrasados (`my_derivada_tres_puntos_extremos`) y `np.gradient` (`derivada_numpy`).
   * **Contexto:** obtención del perfil de desaceleración $a(t) = \frac{dV}{dt}$ y contraste de la desaceleración máxima frente al umbral de confort de $0{,}3g$.

6. **Búsqueda de raíces 1D (`resolver_raiz_con_consenso`)**
   * **Métodos:** Newton-Raphson (`my_newton`), bisección vectorizada (`my_biseccion`) y localización con interpolación lineal local en NumPy (`raices_numpy`).
   * **Contexto:** resolución de $V(t) - V_{\text{taxi}} = 0$ para hallar el instante $t_{\text{taxi}}$ en el que la aeronave alcanza la velocidad objetivo de rodaje ($V_{\text{taxi}} = 15\text{ m/s}$ inicialmente).

---

## Modelado físico y mejoras extra (`[EXTRA]`)

Además de los cálculos base del enunciado, se han incorporado dos mejoras señalizadas explícitamente en el código con la etiqueta `[EXTRA]`:

* **`[EXTRA]` Margen operativo de pista desde Altura de Decisión (`evaluar_margen_pista_total` en `physics.py`):** estima la distancia horizontal de aproximación desde $\text{DH} = 60\text{ m}$ según el ángulo de senda ILS, la suma a la distancia de frenado integrada $S$ y calcula los metros y el porcentaje de margen libre respecto a una longitud de pista de referencia de $2800\text{ m}$.
* **`[EXTRA]` Sliders interactivos en tiempo real (`dashboard_gui.py`):** controles deslizantes asociados a cada fase de vuelo para experimentar con escenarios distintos sobre los valores iniciales del trabajo:
  * **Aproximación:** variación del ruido radar $\sigma_{\text{ILS}}$ (`0.0` a `25.0 m`) con recálculo inmediato de la regresión y el ángulo $\gamma$.
  * **Atmósfera:** variación de la altitud de decisión `DH` (`10.0` a `500.0 m`) con actualización en vivo del viento interpolado y el marcador en el perfil vertical.
  * **Frenado:** variación de la velocidad objetivo de rodaje `V_taxi` (`5.0` a `25.0 m/s`) con recálculo instantáneo de $t_{\text{taxi}}$ y actualización del marcador y el panel lateral.

---

## Dashboard interactivo

La ventana interactiva de `matplotlib` se estructura en:

* **Selector `RadioButtons` (*Fase de Maniobra*):** permite alternar entre **Aproximación**, **Atmósfera** y **Frenado**.
* **Panel numérico lateral (*Resultados Numéricos*):** muestra el reparto de cargas en los trenes de aterrizaje y los indicadores clave de la vista activa.
* **Área gráfica principal:**
  * *Aproximación:* nube de puntos ILS, recta ajustada por regresión lineal y ángulo de descenso.
  * *Atmósfera:* perfil vertical de viento en la capa límite y punto interpolado en la altitud de decisión.
  * *Frenado:* subgráficas sincronizadas de velocidad $V(t)$ (con área integrada $S$ y marcador de $t_{\text{taxi}}$) y aceleración $a(t)$ (con línea de referencia del límite de confort $-0{,}3g$), acompañadas del `Slider` de `V_taxi`.

---

## Criterios técnicos y diseño vectorizado

* Los cálculos matriciales, de regresión, interpolación, integración y derivación están implementados mediante operaciones vectoriales de NumPy y *slicing* sin bucles explícitos `for`/`while`, reservando el bucle `while` exclusivamente para el método iterativo de Newton-Raphson (`my_newton`).
* Las referencias a los *widgets* interactivos (`RadioButtons` y `Slider`) se mantienen ligadas a la figura para garantizar su respuesta continua durante la interacción del usuario.