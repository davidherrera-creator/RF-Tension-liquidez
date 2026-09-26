# Random Forest · Predicción de tensión de liquidez

Aplicación en Streamlit que entrena y evalúa un modelo **Random Forest de clasificación** para predecir la tensión de liquidez (`Tension_Liquidez_bin`: 1 = tensión, 0 = situación normal) a partir de variables de capital de trabajo.

> Base de datos didáctica con datos 100% simulados. No representa una empresa real.

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `app_rf_tension_liquidez.py` | Código de la aplicación |
| `requirements.txt` | Dependencias de Python |
| `README.md` | Este documento |

El archivo Excel **no se incluye** en el repositorio: se sube desde la aplicación en cada sesión.

---

## Datos de entrada

- Formato: Excel (`.xlsx`).
- Hoja obligatoria: **`Datos_Modelo`**.
- Columnas obligatorias: `Tension_Liquidez_bin` y `Sector`.
- Variables excluidas del entrenamiento:

| Variable | Motivo |
|---|---|
| `ID_Observacion` | Identificador |
| `Fecha` | Variable de control |
| `Prob_Tension_Liquidez` | Genera la etiqueta (fuga de información) |
| `Brecha_Caja_90d_MXN` | Información futura; target del modelo de regresión |

---

## Configuración del modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | `RandomForestClassifier` (scikit-learn) en `Pipeline` |
| Variable categórica | `Sector` con `OneHotEncoder(handle_unknown="ignore")` |
| Variables numéricas | Sin escalar |
| Partición | 70% entrenamiento / 30% prueba, estratificada, `random_state=42` |
| Balance de clases | `class_weight=None` |
| Hiperparámetros | `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `random_state=42` |

### Controles interactivos (barra lateral)

1. **Carga de datos:** botón para subir el Excel.
2. **Conjunto de predictores:**
   - *Todas* (21, por defecto).
   - *Solo compuestas:* conserva `CCC_dias`, `NWC_MXN`, `EBITDA_MXN` y retira sus componentes.
   - *Solo componentes:* conserva DSO, DIO, DPO, CxC, Inventarios, CxP, Ventas y Margen; retira las compuestas.
   - *Personalizado:* selección libre.
3. **Umbral de decisión:** de 0.05 a 0.95 (0.50 por defecto).

### Resultados

- **Análisis exploratorio:** distribución del target, tasa de tensión por sector, estadísticos descriptivos, correlación con el target y matriz de correlación.
- **Desempeño en prueba:** accuracy, precisión, recall, F1, ROC-AUC, matriz de confusión, curva ROC y reporte de clasificación por clase.
- **Importancia de variables:** importancia nativa del Random Forest e importancia por permutación (caída de ROC-AUC, 10 repeticiones).

---

## Ejecución local

Requisitos: Python 3.11 o 3.12.

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_rf_tension_liquidez.py
```

La aplicación abre en `http://localhost:8501`.

---

## Publicación en GitHub

**Opción A · Interfaz web**

1. En GitHub, crea un repositorio nuevo (*New repository*). Puede ser público o privado.
2. Selecciona *Add file → Upload files*.
3. Arrastra `app_rf_tension_liquidez.py`, `requirements.txt` y `README.md`.
4. Confirma con *Commit changes*.

**Opción B · Línea de comandos**

```bash
git init
git add app_rf_tension_liquidez.py requirements.txt README.md
git commit -m "App Random Forest tensión de liquidez"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

---

## Despliegue en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Autoriza el acceso a tus repositorios (necesario si el repositorio es privado).
3. Selecciona *Create app → Deploy a public app from GitHub*.
4. Completa:
   - **Repository:** `<usuario>/<repositorio>`
   - **Branch:** `main`
   - **Main file path:** `app_rf_tension_liquidez.py`
   - **App URL:** subdominio de tu elección (opcional).
5. En *Advanced settings*, elige **Python 3.12**.
6. Presiona *Deploy*. La instalación de dependencias tarda unos minutos.

Cada `push` a la rama `main` actualiza la aplicación automáticamente.

---

## Solución de problemas

| Síntoma | Causa probable | Acción |
|---|---|---|
| «El archivo no contiene la hoja Datos_Modelo» | Nombre de hoja distinto | Renombra la hoja a `Datos_Modelo` |
| «Faltan columnas obligatorias» | Falta `Tension_Liquidez_bin` o `Sector` | Revisa encabezados del Excel |
| `ModuleNotFoundError` en el despliegue | `requirements.txt` ausente o fuera de la raíz | Colócalo en la raíz del repositorio |
| Error con el parámetro `width` | Streamlit anterior a 1.50 | Actualiza con `pip install -U streamlit` |
| La app «duerme» tras días sin uso | Política de Community Cloud | Abre la URL y presiona *Wake up* |

---

## Dependencias

```
streamlit>=1.50
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
openpyxl>=3.1
plotly>=5.18
```
