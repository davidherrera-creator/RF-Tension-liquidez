"""
Random Forest — Clasificación de tensión de liquidez
Base didáctica de capital de trabajo (datos simulados)

Ejecutar:  streamlit run app_rf_tension_liquidez.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Configuración fija del modelo
# ---------------------------------------------------------------------------
HOJA = "Datos_Modelo"
TARGET = "Tension_Liquidez_bin"
EXCLUIR = ["ID_Observacion", "Fecha", "Prob_Tension_Liquidez", "Brecha_Caja_90d_MXN"]
CATEGORICAS = ["Sector"]

TEST_SIZE = 0.30
RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Variables compuestas (identidades exactas) y sus componentes
COMPUESTAS = ["CCC_dias", "NWC_MXN", "EBITDA_MXN"]
COMPONENTES = [
    "DSO_dias", "DIO_dias", "DPO_dias",          # CCC = DSO + DIO - DPO
    "CxC_MXN", "Inventarios_MXN", "CxP_MXN",     # NWC = CxC + Inv - CxP
    "Ventas_12M_MXN", "Margen_EBITDA_pct",       # EBITDA = Ventas x Margen
]

st.set_page_config(page_title="RF · Tensión de liquidez", layout="wide")


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(archivo_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(archivo_bytes), sheet_name=HOJA)


def validar(df: pd.DataFrame) -> list[str]:
    faltantes = [c for c in [TARGET] + CATEGORICAS if c not in df.columns]
    return faltantes


def predictores_disponibles(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUIR + [TARGET]]


def construir_preset(preset: str, todas: list[str]) -> list[str]:
    if preset == "Todas":
        return todas
    if preset == "Solo compuestas":
        return [c for c in todas if c not in COMPONENTES]
    if preset == "Solo componentes":
        return [c for c in todas if c not in COMPUESTAS]
    return todas


@st.cache_resource(show_spinner="Entrenando Random Forest…")
def entrenar(archivo_bytes: bytes, features: tuple[str, ...]):
    df = cargar_datos(archivo_bytes)
    X = df[list(features)]
    y = df[TARGET].astype(int)

    cat_cols = [c for c in features if c in CATEGORICAS]
    num_cols = [c for c in features if c not in CATEGORICAS]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )
    modelo = Pipeline([("prep", preproc), ("rf", RandomForestClassifier(**RF_PARAMS))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    modelo.fit(X_tr, y_tr)
    proba_te = modelo.predict_proba(X_te)[:, 1]

    # Importancia nativa (impureza de Gini) sobre columnas transformadas
    nombres = modelo.named_steps["prep"].get_feature_names_out()
    imp_gini = (
        pd.DataFrame({"Variable": nombres,
                      "Importancia": modelo.named_steps["rf"].feature_importances_})
        .sort_values("Importancia", ascending=False)
    )

    # Importancia por permutación sobre variables originales (conjunto de prueba)
    perm = permutation_importance(
        modelo, X_te, y_te, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    imp_perm = (
        pd.DataFrame({"Variable": list(features),
                      "Caída ROC-AUC": perm.importances_mean,
                      "Desv. est.": perm.importances_std})
        .sort_values("Caída ROC-AUC", ascending=False)
    )

    return dict(modelo=modelo, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                proba_te=proba_te, imp_gini=imp_gini, imp_perm=imp_perm)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.header("1 · Datos")
archivo = st.sidebar.file_uploader("Sube el Excel (hoja «Datos_Modelo»)", type=["xlsx"])

st.title("Random Forest · Predicción de tensión de liquidez")
st.caption("Clasificación binaria de `Tension_Liquidez_bin` "
           "(1 = tensión de liquidez · 0 = situación normal). Datos simulados con fines didácticos.")

if archivo is None:
    st.info("Sube el archivo Excel en la barra lateral para iniciar.")
    st.stop()

archivo_bytes = archivo.getvalue()
try:
    df = cargar_datos(archivo_bytes)
except ValueError:
    st.error(f"El archivo no contiene la hoja «{HOJA}».")
    st.stop()

faltantes = validar(df)
if faltantes:
    st.error(f"Faltan columnas obligatorias: {', '.join(faltantes)}")
    st.stop()

todas = predictores_disponibles(df)

st.sidebar.header("2 · Predictores")
preset = st.sidebar.radio(
    "Conjunto de variables",
    ["Todas", "Solo compuestas", "Solo componentes", "Personalizado"],
    help=("Compuestas: CCC, NWC, EBITDA. "
          "Componentes: DSO, DIO, DPO, CxC, Inventarios, CxP, Ventas, Margen."),
)
if preset == "Personalizado":
    features = st.sidebar.multiselect("Selecciona predictores", todas, default=todas)
else:
    features = construir_preset(preset, todas)
    with st.sidebar.expander(f"Variables incluidas ({len(features)})"):
        st.write(features)

if len(features) == 0:
    st.warning("Selecciona al menos un predictor.")
    st.stop()

st.sidebar.header("3 · Umbral de decisión")
umbral = st.sidebar.slider("Probabilidad mínima para clasificar como tensión (1)",
                           0.05, 0.95, 0.50, 0.01)

with st.sidebar.expander("Configuración fija del modelo"):
    st.markdown(
        f"- Partición: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} estratificada, "
        f"`random_state={RANDOM_STATE}`\n"
        + "\n".join(f"- `{k}={v!r}`" for k, v in RF_PARAMS.items())
        + "\n- Excluidas: " + ", ".join(f"`{c}`" for c in EXCLUIR)
    )

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
res = entrenar(archivo_bytes, tuple(features))
y_te, proba = res["y_te"], res["proba_te"]
pred = (proba >= umbral).astype(int)

tab_eda, tab_met, tab_imp = st.tabs(
    ["Análisis exploratorio", "Desempeño del modelo", "Importancia de variables"]
)

# ---------------------------------------------------------------------------
# 7 · Análisis exploratorio
# ---------------------------------------------------------------------------
with tab_eda:
    c1, c2, c3 = st.columns(3)
    c1.metric("Observaciones", f"{len(df):,}")
    c2.metric("Predictores en uso", len(features))
    c3.metric("Tasa de tensión (1)", f"{df[TARGET].mean():.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        dist = df[TARGET].map({0: "0 · Normal", 1: "1 · Tensión"}).value_counts().reset_index()
        dist.columns = ["Clase", "Casos"]
        st.plotly_chart(px.bar(dist, x="Clase", y="Casos", text="Casos",
                               title="Distribución del target"), width="stretch")
    with col_b:
        if "Sector" in df.columns:
            sec = df.groupby("Sector")[TARGET].agg(["count", "mean"]).reset_index()
            sec.columns = ["Sector", "Casos", "Tasa de tensión"]
            fig = px.bar(sec, x="Sector", y="Tasa de tensión", text=sec["Tasa de tensión"].map("{:.0%}".format),
                         hover_data=["Casos"], title="Tasa de tensión por sector")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    st.subheader("Estadísticos descriptivos")
    num_feats = [c for c in features if c not in CATEGORICAS]
    st.dataframe(df[num_feats].describe().T.style.format("{:,.3f}"), width="stretch")

    st.subheader("Correlación con el target")
    corr_t = (df[num_feats + [TARGET]].corr()[TARGET].drop(TARGET)
              .sort_values().reset_index())
    corr_t.columns = ["Variable", "Correlación"]
    fig = px.bar(corr_t, x="Correlación", y="Variable", orientation="h",
                 color="Correlación", color_continuous_scale="RdBu_r", range_color=[-1, 1])
    fig.update_layout(height=max(350, 22 * len(corr_t)))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Matriz de correlación entre predictores")
    corr = df[num_feats].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=max(450, 28 * len(num_feats)))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 1, 2, 3, 6 · Desempeño
# ---------------------------------------------------------------------------
with tab_met:
    st.caption(f"Conjunto de prueba: {len(y_te)} observaciones · umbral = {umbral:.2f}")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{accuracy_score(y_te, pred):.3f}")
    m[1].metric("Precisión", f"{precision_score(y_te, pred, zero_division=0):.3f}")
    m[2].metric("Recall", f"{recall_score(y_te, pred, zero_division=0):.3f}")
    m[3].metric("F1", f"{f1_score(y_te, pred, zero_division=0):.3f}")
    m[4].metric("ROC-AUC", f"{roc_auc_score(y_te, proba):.3f}",
                help="Independiente del umbral.")

    col_a, col_b = st.columns(2)
    with col_a:
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        etiquetas = ["0 · Normal", "1 · Tensión"]
        fig = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicción", y="Real", color="Casos"),
                        title="Matriz de confusión")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        fpr, tpr, thr = roc_curve(y_te, proba)
        idx = int(np.argmin(np.abs(thr - umbral)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"RF (AUC = {roc_auc_score(y_te, proba):.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash"), name="Azar"))
        fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode="markers",
                                 marker=dict(size=11), name=f"Umbral {umbral:.2f}"))
        fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos (recall)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Reporte de clasificación por clase")
    rep = classification_report(y_te, pred, target_names=["0 · Normal", "1 · Tensión"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"), width="stretch")

# ---------------------------------------------------------------------------
# 4, 5 · Importancia de variables
# ---------------------------------------------------------------------------
with tab_imp:
    col_a, col_b = st.columns(2)
    with col_a:
        g = res["imp_gini"].sort_values("Importancia")
        fig = px.bar(g, x="Importancia", y="Variable", orientation="h",
                     title="Importancia nativa (reducción de impureza)")
        fig.update_layout(height=max(450, 24 * len(g)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Calculada en entrenamiento; `Sector` aparece desglosado por categoría.")
    with col_b:
        p = res["imp_perm"].sort_values("Caída ROC-AUC")
        fig = px.bar(p, x="Caída ROC-AUC", y="Variable", orientation="h",
                     error_x="Desv. est.", title="Importancia por permutación (prueba)")
        fig.update_layout(height=max(450, 24 * len(p)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Caída promedio del ROC-AUC al permutar cada variable (10 repeticiones).")
