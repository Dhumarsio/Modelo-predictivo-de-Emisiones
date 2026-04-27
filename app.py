import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Emisiones en Tambos", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #ffffff; color: #0c1a2e; }
    .block-container { padding-top: 1.2rem !important; }
    h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; font-weight: 800 !important; color: #0c1a2e !important; letter-spacing: -0.02em; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #f8fafc; padding: 0 8px; border-bottom: 1px solid #e2e8f0; border-radius: 0; }
    .stTabs [data-baseweb="tab"] { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.86rem; padding: 10px 18px; border-radius: 0; color: #94a3b8; background: transparent; border-bottom: 2px solid transparent; margin-bottom: -1px; }
    .stTabs [aria-selected="true"] { background: transparent !important; color: #0ea5e9 !important; border-bottom: 2px solid #0ea5e9 !important; }
    .metric-card { background: #f0f9ff; border-radius: 12px; border-top: 3px solid #0ea5e9; padding: 16px 16px 14px; text-align: left; }
    .metric-card .label { font-size: 0.68rem; color: #0ea5e9; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 5px; }
    .metric-card .value { font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; color: #0c1a2e; letter-spacing: -0.02em; line-height: 1; }
    .metric-card .delta { font-size: 0.68rem; margin-top: 4px; font-weight: 600; color: #64748b; }
    .result-box { background: #f0f9ff; border-radius: 12px; border-top: 3px solid #0ea5e9; padding: 20px 18px 18px; color: #0c1a2e; text-align: left; }
    .result-box .big-number { font-family: 'Outfit', sans-serif; font-size: 2.6rem; font-weight: 800; color: #0c1a2e; letter-spacing: -0.02em; line-height: 1; }
    .result-box .rlabel { font-size: 0.68rem; color: #0ea5e9; text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px; font-weight: 700; }
    .result-box .sub { font-size: 0.76rem; color: #64748b; margin-top: 5px; }
    .info-banner { background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 0 8px 8px 0; padding: 13px 16px; margin: 14px 0; font-size: 0.86rem; color: #0c4a6e; line-height: 1.5; }
    .stButton>button { background: #0ea5e9; color: white; font-family: 'Outfit', sans-serif; font-weight: 700; border: none; border-radius: 8px; padding: 12px 32px; font-size: 1rem; width: 100%; }
    .stButton>button:hover { background: #0284c7; }
    section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

PRECIO_CARBONO_EUR = 16
EUR_A_USD = 1.08
PRECIO_CARBONO_USD = PRECIO_CARBONO_EUR * EUR_A_USD
TARGET = 'Total Establecimiento TNCO2eq / año'
EXCLUIR_TARGET = [TARGET, 'Total leche kgCO2eq/lt', 'TNCO2eq_por_litro', 'LCGP.vo', 'LCG', 'LCGP.HA', 'LCGP']
EXCLUIR_BAJA_VAR = ['PV VO', "%GB"]
EXCLUIR_IDS = ['ID', 'Provincia', 'Localidad', 'CUENCA LECHERA']
VARS_MODELO_FINAL = ['VO', 'kg LCGP']

COLORES = {
    'primary': '#0ea5e9', 'secondary': '#38bdf8', 'light': '#7dd3fc',
    'pale': '#e0f2fe', 'dark': '#0284c7', 'accent': '#f97316',
    'green': '#22c55e', 'red': '#ef4444', 'bg': '#f0f9ff',
    'grid': '#e2e8f0', 'text': '#0c1a2e',
}

PLOTLY_TEMPLATE = dict(layout=dict(
    font=dict(family='Outfit, sans-serif', color=COLORES['text'], size=12),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    colorway=[COLORES['primary'], COLORES['dark'], COLORES['secondary'], COLORES['light']],
    xaxis=dict(showgrid=True, gridcolor=COLORES['grid'], linecolor='#cbd5e1', zeroline=False, tickfont=dict(size=11, family='Outfit')),
    yaxis=dict(showgrid=True, gridcolor=COLORES['grid'], linecolor='#cbd5e1', zeroline=False, tickfont=dict(size=11, family='Outfit')),
    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Outfit', bordercolor='#e2e8f0'),
))

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_excel(archivo)
    if 'Provincia' in df.columns:
        df['Provincia'] = df['Provincia'].str.strip().str.title()
    le = LabelEncoder()
    if 'Sistema alimentación' in df.columns:
        df['Sistema alimentación'] = le.fit_transform(df['Sistema alimentación'].astype(str))
    df['Costo_USD'] = df[TARGET] * PRECIO_CARBONO_USD
    df['Costo_EUR'] = df[TARGET] * PRECIO_CARBONO_EUR
    return df

@st.cache_resource
def entrenar_modelos(df):
    excluir = EXCLUIR_TARGET + EXCLUIR_BAJA_VAR + EXCLUIR_IDS + ['Costo_USD', 'Costo_EUR']
    feature_cols = [c for c in df.columns if c not in excluir + [TARGET]]
    X_full = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df[TARGET]
    X_final = X_full[VARS_MODELO_FINAL]
    X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(X_full, y, test_size=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_final, y, test_size=0.2, random_state=42)
    dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=42).fit(X_tr_full, y_tr_full)
    rf_shap = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(X_tr_full, y_tr_full)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42).fit(X_tr_full, y_tr_full)
    rf_final = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    lr = LinearRegression().fit(X_tr, y_tr)
    cv_dt = cross_val_score(dt, X_full, y, cv=5, scoring='r2')
    cv_rf_shap = cross_val_score(rf_shap, X_full, y, cv=5, scoring='r2')
    cv_gb = cross_val_score(gb, X_full, y, cv=5, scoring='r2')
    cv_rf = cross_val_score(rf_final, X_final, y, cv=5, scoring='r2')
    cv_lr = cross_val_score(lr, X_final, y, cv=5, scoring='r2')
    return {
        'df': df, 'X_full': X_full, 'X_final': X_final, 'X_shap': X_full, 'y': y,
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'X_tr_full': X_tr_full, 'X_te_full': X_te_full, 'y_tr_full': y_tr_full, 'y_te_full': y_te_full,
        'lr': lr, 'rf': rf_final, 'rf_shap': rf_shap, 'dt': dt, 'gb': gb,
        'cv_lr': cv_lr, 'cv_rf': cv_rf, 'cv_rf_shap': cv_rf_shap, 'cv_dt': cv_dt, 'cv_gb': cv_gb,
        'feature_cols': feature_cols
    }

st.markdown("""
<div style="border-bottom: 3px solid #0ea5e9; padding-bottom: 14px; margin-bottom: 22px;">
    <div style="font-size:0.70rem; color:#0ea5e9; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">🐄 Tambos · IPCC 2019 · Argentina</div>
    <div style="font-size:1.9rem; font-weight:800; color:#0c1a2e; letter-spacing:-0.02em; line-height:1.1; font-family:'Outfit',sans-serif;">Análisis de Emisiones en Tambos</div>
    <div style="font-size:0.85rem; color:#64748b; margin-top:5px; font-weight:400;">Modelo predictivo de huella de carbono y costo ambiental · Argentina</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Cargar Dataset")
    archivo = st.file_uploader("Subí el archivo Excel", type=["xlsx"])
    st.markdown("---")
    st.markdown("### ⚙️ Parámetros")
    st.markdown(f"**Precio carbono:** €{PRECIO_CARBONO_EUR}/tCO₂eq")
    st.markdown(f"**Tipo de cambio:** 1 EUR = {EUR_A_USD} USD")
    st.markdown(f"**Precio en USD:** ${PRECIO_CARBONO_USD:.2f}/tCO₂eq")
    st.markdown("---")
    st.markdown("*Diplomatura en Ciencia de Datos con R y Python*")

if archivo is None:
    st.info("👈 Subí el archivo desde el panel izquierdo para comenzar.")
    st.stop()

df_raw = cargar_datos(archivo)
m = entrenar_modelos(df_raw)

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Introducción", "📊 EDA", "🌳 Modelos Iniciales",
    "🔍 SHAP", "📈 Modelo Final", "🗂️ Clusters", "🧮 Predictor",
])

# ════════ TAB 0 — INTRODUCCIÓN ════════
with tab0:
    st.markdown("## El Problema")
    st.markdown("""<div class="info-banner">Este trabajo fue desarrollado en el marco de la <strong>Diplomatura en Ciencia de Datos con R y Python</strong> como proyecto final integrador. El objetivo es aplicar herramientas de análisis exploratorio, aprendizaje automático y visualización interactiva sobre un problema real del sector agropecuario argentino.</div>""", unsafe_allow_html=True)

    col_txt, col_stats = st.columns([3, 2])
    with col_txt:
        st.markdown("### 🌍 Contexto Global")
        st.markdown("""El cambio climático es uno de los desafíos más urgentes de la agenda global. El sector agropecuario representa aproximadamente el **14% de las emisiones mundiales de gases de efecto invernadero (GEI)**, y dentro de él, la ganadería bovina es la principal fuente de emisiones debido a la **fermentación entérica** (metano producido por los rumiantes durante la digestión) y al **manejo del estiércol**.\n\nEn Argentina, la producción de leche ocupa un lugar estratégico tanto en la economía como en el territorio. El país cuenta con aproximadamente **10.000 tambos activos**, concentrados principalmente en la región pampeana, que producen alrededor de **11.000 millones de litros de leche al año**.""")
        st.markdown("### 🐄 ¿Qué es un tambo?")
        st.markdown("""Un tambo es un establecimiento dedicado a la producción de leche bovina. A diferencia de los feedlots, los tambos argentinos se basan mayormente en **sistemas pastoriles**, donde las vacas en ordeñe (VO) se alimentan principalmente de pasturas implantadas, complementadas con reservas forrajeras (heno, silaje) y concentrados energéticos.\n\nLas principales fuentes de emisiones de un tambo son:\n- **Fermentación entérica:** el metano (CH₄) producido en el rumen de las vacas. Es la fuente dominante.\n- **Gestión del estiércol:** emisiones de CH₄ y N₂O del manejo de las heces y orina.\n- **Fertilización nitrogenada:** el óxido nitroso (N₂O) liberado al aplicar fertilizantes al suelo.\n- **Energía y combustibles:** emisiones de CO₂ por el uso de electricidad y gasoil en el establecimiento.""")

    with col_stats:
        st.markdown("### 📌 Dataset")
        for val, lbl in [
            (f"{df_raw.shape[0]}", "Tambos relevados"),
            (f"{df_raw.shape[1] - 2}", "Variables originales"),
            (f"{df_raw[TARGET].mean():,.0f}", "TNCO₂eq/año promedio"),
            (f"{df_raw['VO'].mean():,.0f}", "Vacas en ordeñe promedio"),
            (f"{df_raw['Provincia'].nunique()}", "Provincias"),
        ]:
            st.markdown(f'<div class="metric-card" style="margin-bottom:10px"><div class="label">{lbl}</div><div class="value" style="font-size:1.8rem">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_met, col_obj = st.columns(2)
    with col_met:
        st.markdown("### 📐 Metodología de cálculo — IPCC 2019")
        st.markdown("""Las emisiones de cada tambo fueron calculadas siguiendo las **directrices del IPCC 2019**, el estándar internacional de referencia para la estimación de GEI en el sector agropecuario.\n\nLas emisiones se expresan en **toneladas de CO₂ equivalente por año (TNCO₂eq/año)**, unificando metano, óxido nitroso y dióxido de carbono mediante sus potenciales de calentamiento global (GWP100):\n- CH₄ = 27 CO₂eq\n- N₂O = 273 CO₂eq\n\nEl dataset fue construido extrayendo variables productivas y de manejo de 125 tambos reales de Argentina utilizados para calcular la huella de carbono según la metodología del IPCC.""")
    with col_obj:
        st.markdown("### 🎯 Objetivos del Proyecto")
        st.markdown("""**1. Explorar y comprender el dataset**\nAnalizar la distribución de emisiones entre tambos, las diferencias geográficas y las correlaciones entre variables productivas y huella de carbono.\n\n**2. Identificar perfiles de tambos**\nAgrupar los establecimientos en clusters según sus características para identificar tipos de sistemas productivos y sus impactos ambientales asociados.\n\n**3. Construir un modelo predictivo**\nEntrenar y comparar modelos de machine learning para predecir las emisiones totales de un tambo a partir de variables simples y fácilmente relevables.\n\n**4. Cuantificar el costo ambiental**\nTraducir las emisiones predichas a un costo económico concreto, utilizando como referencia la **tasa de carbono danesa (€16/tCO₂eq)**.""")

    st.markdown("---")
    st.markdown("### 💡 ¿Por qué la tasa de carbono danesa?")
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.markdown(f'<div class="metric-card"><div class="label">Tasa de referencia</div><div class="value">€16</div><div class="delta">por tCO₂eq · Dinamarca 2023</div></div>', unsafe_allow_html=True)
    col_c2.markdown(f'<div class="metric-card"><div class="label">Equivalente en USD</div><div class="value">${PRECIO_CARBONO_USD:.2f}</div><div class="delta">tipo de cambio 1 EUR = {EUR_A_USD} USD</div></div>', unsafe_allow_html=True)
    col_c3.markdown(f'<div class="metric-card"><div class="label">Costo promedio por tambo</div><div class="value">${df_raw["Costo_USD"].mean():,.0f}</div><div class="delta">USD / año (dataset)</div></div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-banner">Dinamarca implementó en 2023 un impuesto al carbono específico para el sector agropecuario, siendo uno de los primeros países del mundo en gravar directamente las emisiones de la ganadería. Se utiliza como <strong>benchmark internacional</strong> para poner en perspectiva el impacto ambiental del sector.</div>""", unsafe_allow_html=True)

# ════════ TAB 1 — EDA ════════
with tab1:
    st.markdown("## Análisis Exploratorio de Datos")
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, lbl, unit) in zip([col1, col2, col3, col4], [
        (f"{df_raw[TARGET].mean():,.0f}", "Emisiones promedio", "TNCO₂eq/año"),
        (f"{df_raw[TARGET].min():,.0f}", "Emisión mínima", "TNCO₂eq/año"),
        (f"{df_raw[TARGET].max():,.0f}", "Emisión máxima", "TNCO₂eq/año"),
        (f"{df_raw['VO'].mean():,.0f}", "Vacas en ordeñe promedio", "cabezas"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div><div class="delta">{unit}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ Vista del Dataset")
    st.markdown("""<div class="info-banner">El dataset fue construido a partir de la herramienta de cálculo de emisiones del <strong>IPCC 2019</strong>, relevando datos productivos, de manejo y ambientales de <strong>125 tambos</strong> de Argentina. La variable objetivo es el total de emisiones del establecimiento en <strong>TNCO₂eq/año</strong>.</div>""", unsafe_allow_html=True)

    GRUPOS = {
        '🏷️ Identificación': ['ID', 'Provincia', 'Localidad', 'CUENCA LECHERA'],
        '🐄 Rodeo & Producción': ['VO', 'VS', 'VT', 'VACAS ADULTA', 'Carga VT', ' % VACA ORD.', 'LCG', 'PV VO', 'lts', 'kg/d', '%PC', '% PV', 'LCGP', 'LCGP.vo', 'LCGP.HA', 'kg LCGP', 'CMS', 'CMS Vo'],
        '🌾 Alimentación VO': ['Sistema alimentación', 'CMS pastura', 'CMS pastura Vo', 'BALANCEADO', 'AFRECHILLO', 'ALGODÓN', 'PELLET SOJA', 'MAIZ', 'MAIZ GH', 'SORGO', 'SORGO GH', 'OTROS', 'HENO', 'Silo pastura', 'Heno pastura', 'Heno otro', 'Silo verdeo', 'Silo maíz', 'Silo Sorgo\nconsumido año kg tal cual', 'Silo Girasol\nconsumido año kg tal cual', 'Maiz Gr Humedo\nTn año', 'Sorgo Gr HUmedo\nTn año', 'Kg MS concentrado / VO / dia - promedio anual', '% FDN dieta VO', '% PB DIETA VO', '% DIG VO'],
        '🌾 Alimentación VS': ['concentrado/VS.dia', 'Reservas.VS.d', 'CMSvs', 'CMSpast VS', '% FDN dieta VS', '% PB DIETA VS', '% DIG VS'],
        '🗺️ Superficie & Suelo': ['SUPERFICIE TOTAL TAMBO (Has)', 'Sup VACA ADULTA (Has)', 'RECRIA         (Has)', 'AGRICULTURA\n(Has)', 'GANADERIA\n(Has)', 'Pastura (ha)', 'Pastura tambo (ha)', 'VI (ha)', 'VV (ha)', 'Verdeo inv reserva (ha)', 'Maíz para silaje (ha)', 'Sorgo silaje (ha)', 'Girasol  silaje (ha)', 'Maíz grano (ha)', 'Sorgo grano (ha)'],
        '🛒 Compras de insumos': ['Compra balanceado', 'Compra afrechillo', 'Compra algodón', 'Pellet soja', 'Maíz comprado', 'Sorgo comprado', 'Otros'],
        '⚗️ Fertilización & Energía': ['N_UREA', 'AMONIACAL', 'NPK', 'kgN total', 'kgN.ha', 'Consumo Gasoil', 'Energía Total '],
        '🏭 Emisiones': [TARGET, 'Total leche kgCO2eq/lt'],
    }

    col_info1, col_info2 = st.columns([1, 2])
    with col_info1:
        st.markdown("**Variables por categoría**")
        resumen_rows = []
        total_clasificadas = 0
        for grupo, vars_g in GRUPOS.items():
            presentes = [v for v in vars_g if v in df_raw.columns]
            resumen_rows.append({'Categoría': grupo, 'N° variables': len(presentes)})
            total_clasificadas += len(presentes)
        st.dataframe(pd.DataFrame(resumen_rows), use_container_width=True, hide_index=True)
        st.markdown(f"<small style='color:#64748b'>Total clasificadas: <strong>{total_clasificadas}</strong> · Total en dataset: <strong>{df_raw.shape[1]}</strong> · Tambos: <strong>{df_raw.shape[0]}</strong></small>", unsafe_allow_html=True)

    with col_info2:
        st.markdown("**Vista previa — primeras filas del dataset**")
        prov_opts = ['Todas'] + sorted(df_raw['Provincia'].dropna().unique().tolist())
        prov_sel = st.selectbox("Filtrar por provincia", prov_opts, label_visibility='collapsed')
        df_preview = df_raw if prov_sel == 'Todas' else df_raw[df_raw['Provincia'] == prov_sel]
        cols_show = [c for c in ['ID', 'Provincia', 'Localidad', 'VO', 'VS', 'VT', 'lts', '%PC', 'CMS', 'CMS pastura', 'SUPERFICIE TOTAL TAMBO (Has)', 'kgN total', 'Consumo Gasoil', 'Energía Total ', TARGET] if c in df_preview.columns]
        st.dataframe(df_preview[cols_show].head(10).reset_index(drop=True).round(2), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Distribución y Geografía de Emisiones")
    st.markdown("""<div class="info-banner">El primer paso es entender <strong>cómo se distribuyen las emisiones totales</strong> entre los tambos del dataset. Una distribución sesgada a la derecha indicaría que la mayoría de los tambos emite poco pero un grupo reducido concentra emisiones muy altas. El gráfico por provincia permite identificar diferencias geográficas sistemáticas.</div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Distribución de Emisiones Totales")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_raw[TARGET], nbinsx=20, marker_color=COLORES['primary'], marker_line_color='white', marker_line_width=1, opacity=0.85, name='Tambos', hovertemplate='Rango: %{x}<br>Frecuencia: %{y}<extra></extra>'))
        fig.add_vline(x=df_raw[TARGET].mean(), line_dash='dash', line_color=COLORES['light'], line_width=2, annotation_text=f"Promedio: {df_raw[TARGET].mean():,.0f}", annotation_position="top right", annotation_font_color=COLORES['primary'])
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='TNCO₂eq/año', yaxis_title='Frecuencia', showlegend=False, height=360, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Emisiones Promedio por Provincia")
        prov_data = df_raw.groupby('Provincia')[TARGET].agg(['mean','count']).reset_index()
        prov_data.columns = ['Provincia', 'Emisiones', 'N']
        prov_data = prov_data.sort_values('Emisiones')
        fig = go.Figure(go.Bar(x=prov_data['Emisiones'], y=prov_data['Provincia'], orientation='h', marker_color=COLORES['secondary'], marker_line_color='white', marker_line_width=1, customdata=prov_data['N'].values, hovertemplate='<b>%{y}</b><br>Emisiones promedio: %{x:,.0f} TNCO₂eq/año<br>Tambos: %{customdata}<extra></extra>'))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='Emisiones promedio (TNCO₂eq/año)', yaxis_title='', height=360, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Estadísticas Descriptivas")
    st.markdown("""<div class="info-banner">La tabla resume las principales variables del dataset: media, desvío estándar, mínimo, cuartiles y máximo. Es útil para detectar outliers, asimetría y el rango de variación de cada variable.</div>""", unsafe_allow_html=True)
    vars_eda = [v for v in ['VO', 'VS', 'VT', 'VACAS ADULTA', 'lts', 'kg/d', '%PC', 'CMS', 'CMS pastura', 'concentrado/VS.dia', 'SUPERFICIE TOTAL TAMBO (Has)', 'Pastura (ha)', 'kgN total', 'kgN.ha', 'Consumo Gasoil', 'Energía Total ', TARGET, 'Total leche kgCO2eq/lt'] if v in df_raw.columns]
    st.dataframe(df_raw[vars_eda].describe().round(2), use_container_width=True)

    # FIX 1: bloque if completo para la nota de ceros
    n_gasoil_0 = int((df_raw['Consumo Gasoil'] == 0).sum()) if 'Consumo Gasoil' in df_raw.columns else 0
    n_energia_0 = int((df_raw['Energía Total '] == 0).sum()) if 'Energía Total ' in df_raw.columns else 0
    if n_gasoil_0 > 0 or n_energia_0 > 0:
        st.markdown(f"""<div class="info-banner">⚠️ <strong>Valores cero en variables de energía:</strong> se detectaron <strong>{n_gasoil_0} tambos</strong> con Consumo Gasoil = 0 y <strong>{n_energia_0} tambos</strong> con Energía Total = 0. Es probable que en estos casos el dato no fue relevado y se registró como 0 en lugar de valor ausente (NaN), lo que explica el mínimo de 0 en esas columnas. Estos ceros <strong>no afectan el modelo predictivo</strong> ya que estas variables no entraron al entrenamiento.</div>""", unsafe_allow_html=True)

    st.markdown("### 🔗 Correlación con las Emisiones Totales")
    st.markdown("""<div class="info-banner">Se calculó la <strong>correlación de Pearson en valor absoluto</strong> entre cada variable numérica y el target. Valores cercanos a 1 indican una relación lineal fuerte. Las variables con mayor correlación serán las candidatas más naturales para el modelo predictivo.</div>""", unsafe_allow_html=True)
    num_cols = df_raw.select_dtypes(include='number').columns
    excluir_corr = [TARGET, 'Costo_USD', 'Costo_EUR'] + EXCLUIR_TARGET
    corr_target = (df_raw[num_cols].corr()[TARGET].drop([c for c in excluir_corr if c in df_raw[num_cols].columns], errors='ignore').abs().sort_values(ascending=False).head(20).reset_index())
    corr_target.columns = ['Variable', 'Correlación']
    corr_target['color'] = corr_target['Correlación'].apply(lambda v: f'rgba({int(14 + v*20)},{int(165 - v*60)},{int(233 - v*80)},{0.6 + v*0.4})')
    fig = go.Figure(go.Bar(x=corr_target['Variable'], y=corr_target['Correlación'], marker_color=corr_target['color'].tolist(), marker_line_color='white', marker_line_width=1, hovertemplate='<b>%{x}</b><br>|Correlación| con emisiones: %{y:.3f}<extra></extra>'))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], yaxis_title='|Correlación con TNCO₂eq/año|', xaxis_tickangle=-45, height=420, margin=dict(l=20, r=20, t=20, b=120))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="info-banner">⚠️ <strong>Nota sobre PV (Peso Vivo):</strong> esta variable aparece con alta correlación pero tiene variabilidad prácticamente nula, ya que el IPCC asigna un valor estabulado fijo. No aporta información real al modelo y fue excluida del entrenamiento.</div>""", unsafe_allow_html=True)

    st.markdown("### 🏭 Ranking de Emisiones por Tambo")
    st.markdown("""<div class="info-banner">Este gráfico ordena los tambos de mayor a menor emisión total. La codificación por color permite identificar los tres tercios del dataset: <strong>alto emisor</strong> (top 33%), <strong>emisor medio</strong> y <strong>bajo emisor</strong> (bottom 33%). Pasando el cursor sobre cada barra se pueden ver el ID, la provincia, las vacas en ordeñe y las emisiones exactas.</div>""", unsafe_allow_html=True)
    df_sorted = df_raw[['ID', 'Provincia', 'VO', TARGET]].sort_values(TARGET, ascending=False).reset_index(drop=True)
    q33 = df_raw[TARGET].quantile(0.33)
    q66 = df_raw[TARGET].quantile(0.66)
    categorias = ['Alto emisor' if v > q66 else 'Emisor medio' if v > q33 else 'Bajo emisor' for v in df_sorted[TARGET]]
    fig = go.Figure()
    for cat, color in [('Alto emisor (top 33%)', COLORES['primary']), ('Emisor medio', COLORES['light']), ('Bajo emisor (bottom 33%)', COLORES['pale'])]:
        mask = [c == cat.split(' (')[0] for c in categorias]
        fig.add_trace(go.Bar(x=df_sorted.index[mask], y=df_sorted[TARGET][mask], name=cat, marker_color=color, marker_line_color='white', marker_line_width=0.8, hovertemplate='<b>ID %{customdata[0]}</b><br>Prov: %{customdata[1]}<br>VO: %{customdata[2]}<br>Emisiones: %{y:,.0f} TNCO₂eq/año<extra></extra>', customdata=df_sorted[['ID', 'Provincia', 'VO']][mask].values))
    fig.add_hline(y=df_raw[TARGET].mean(), line_dash='dash', line_color='#334155', line_width=1.5, annotation_text=f"Promedio: {df_raw[TARGET].mean():,.0f}", annotation_position='top right')
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], barmode='overlay', xaxis_title='Tambos (ordenados por emisiones)', yaxis_title='TNCO₂eq/año', xaxis_showticklabels=False, height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=20, r=20, t=40, b=40), bargap=0.15)
    st.plotly_chart(fig, use_container_width=True)

# ════════ TAB 2 — MODELOS INICIALES ════════
with tab2:
    st.markdown("## Modelos Iniciales")
    n_vars_modelo = len(m['feature_cols'])
    st.markdown(f"""<div class="info-banner">Como punto de partida se entrenaron <strong>tres modelos</strong> utilizando las <strong>{n_vars_modelo} variables</strong> disponibles tras la limpieza del dataset (de las {df_raw.shape[1]} originales, se excluyeron variables relacionadas al target, de baja variabilidad y de identificación): Árbol de Decisión, Random Forest y Gradient Boosting.</div>""", unsafe_allow_html=True)

    st.markdown("### 📐 Comparativa de Rendimiento")
    st.markdown("""<div class="info-banner">Para cada modelo se muestran dos métricas de R²:<ul style="margin-top:6px; margin-bottom:0"><li><strong>R² CV (5-fold):</strong> promedio de 5 particiones distintas del dataset, más confiable y menos sujeto al azar de un único split.</li><li><strong>R² Test:</strong> medido sobre el 20% de datos que el modelo nunca vio durante el entrenamiento.</li></ul>Ambos deberían ser similares. Si el R² test es mucho menor que el CV, hay sobreajuste.</div>""", unsafe_allow_html=True)

    modelos_info = [("🌳 Árbol de Decisión", m['cv_dt'], m['dt']), ("🌲 Random Forest", m['cv_rf_shap'], m['rf_shap']), ("⚡ Gradient Boosting", m['cv_gb'], m['gb'])]
    r2_tests = [r2_score(m['y_te_full'], modelo.predict(m['X_te_full'])) for _, _, modelo in modelos_info]

    col1, col2, col3 = st.columns(3)
    for col, (nombre, cv, _), r2_t in zip([col1, col2, col3], modelos_info, r2_tests):
        col.markdown(f'<div class="metric-card"><div class="label">{nombre}</div><div class="value">{cv.mean():.3f}</div><div class="delta">R² CV (5-fold) &nbsp;±&nbsp;{cv.std():.3f}</div><div class="delta" style="margin-top:2px; color:#0284c7">R² Test = {r2_t:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Importancia de Variables por Modelo")
    st.markdown("""<div class="info-banner">Cada modelo mide la importancia de las variables por reducción de impureza. Si los tres modelos coinciden en las mismas variables importantes, eso es una señal sólida de que esas variables realmente predicen las emisiones.</div>""", unsafe_allow_html=True)

    def norm(s):
        total = s.sum()
        return (s / total * 100).round(2) if total > 0 else s

    imp_dt = norm(pd.Series(m['dt'].feature_importances_, index=m['X_shap'].columns)).sort_values().tail(15)
    imp_rf = norm(pd.Series(m['rf_shap'].feature_importances_, index=m['X_shap'].columns)).sort_values().tail(15)
    imp_gb = norm(pd.Series(m['gb'].feature_importances_, index=m['X_shap'].columns)).sort_values().tail(15)

    fig = make_subplots(rows=1, cols=3, subplot_titles=['Árbol de Decisión', 'Random Forest', 'Gradient Boosting'])
    for i, (imp, color) in enumerate([(imp_dt, COLORES['secondary']), (imp_rf, COLORES['primary']), (imp_gb, COLORES['dark'])], 1):
        fig.add_trace(go.Bar(x=imp.values, y=imp.index, orientation='h', marker_color=color, marker_line_width=0, showlegend=False, hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}%<extra></extra>'), row=1, col=i)
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=480, margin=dict(l=20, r=20, t=50, b=20))
    for i in range(1, 4):
        fig.update_xaxes(title_text='Importancia (%)', row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Real vs Predicho")
    st.markdown("""<div class="info-banner">Visualización del set de test (20% del dataset). Cada punto es un tambo: el eje X es la emisión real y el eje Y la predicha por el modelo. Cuanto más cerca de la diagonal roja, mejor.</div>""", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3, subplot_titles=['placeholder', 'placeholder', 'placeholder'])
    for i, ((nombre, cv, modelo), r2_t) in enumerate(zip(modelos_info, r2_tests), 1):
        y_pred = modelo.predict(m['X_te_full'])
        lims = [min(m['y_te_full'].min(), y_pred.min()), max(m['y_te_full'].max(), y_pred.max())]
        fig.add_trace(go.Scatter(x=m['y_te_full'], y=y_pred, mode='markers', marker=dict(color=COLORES['primary'], size=7, opacity=0.7, line=dict(color='white', width=1)), showlegend=False, hovertemplate='Real: %{x:,.0f}<br>Predicho: %{y:,.0f}<extra></extra>', name=nombre), row=1, col=i)
        fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', line=dict(color='red', dash='dash', width=1.5), showlegend=False), row=1, col=i)
        fig.layout.annotations[i-1].text = f'{nombre}<br><span style="font-size:11px">R² CV={cv.mean():.3f} · R² Test={r2_t:.3f}</span>'
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=420, margin=dict(l=20, r=20, t=60, b=20))
    for i in range(1, 4):
        fig.update_xaxes(title_text='Real', row=1, col=i)
        fig.update_yaxes(title_text='Predicho', row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)

# ════════ TAB 3 — SHAP ════════
with tab3:
    st.markdown("## Análisis SHAP")
    st.markdown("""<div class="info-banner">SHAP (SHapley Additive exPlanations) permite entender cuánto aporta cada variable a la predicción de cada tambo individualmente, basándose en la teoría de juegos colaborativos de Shapley. Este análisis se corre sobre el Random Forest entrenado con todas las variables, y nos permite identificar cuáles son realmente importantes.</div>""", unsafe_allow_html=True)

    explainer = shap.TreeExplainer(m['rf_shap'])
    shap_values = explainer.shap_values(m['X_shap'])
    ev = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
    st.markdown(f"**Valor base (promedio emisiones):** {ev:,.1f} TNCO2eq/año")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Importancia Media Absoluta")
        shap.summary_plot(shap_values, m['X_shap'], plot_type='bar', show=False)
        st.pyplot(plt.gcf())
        plt.close()
    with col_b:
        st.markdown("### Beeswarm – Dirección del Efecto")
        shap.summary_plot(shap_values, m['X_shap'], show=False)
        st.pyplot(plt.gcf())
        plt.close()

    st.markdown("### Waterfall – Descomposición por Tambo")
    idx_max = m['y'].values.argmax()
    idx_min = m['y'].values.argmin()
    col1, col2 = st.columns(2)
    for col, idx, label in [(col1, idx_max, 'Mayor emisor'), (col2, idx_min, 'Menor emisor')]:
        with col:
            st.markdown(f"#### Tambo {label} (ID={df_raw.iloc[idx]['ID']})")
            exp = shap.Explanation(values=shap_values[idx], base_values=ev, data=m['X_shap'].iloc[idx], feature_names=m['X_shap'].columns.tolist())
            shap.plots.waterfall(exp, max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.close()

    st.markdown("### Ranking SHAP")
    shap_tabla = pd.DataFrame({'SHAP medio absoluto': np.abs(shap_values).mean(axis=0), 'SHAP medio con signo': shap_values.mean(axis=0)}, index=m['X_shap'].columns).sort_values('SHAP medio absoluto', ascending=False)
    shap_tabla['Dirección'] = shap_tabla['SHAP medio con signo'].apply(lambda x: '↑ Sube emisiones' if x > 0 else '↓ Baja emisiones')
    st.dataframe(shap_tabla.round(4), use_container_width=True)

# ════════ TAB 4 — MODELO FINAL ════════
with tab4:
    st.markdown("## Modelo Final — Regresión Lineal")
    st.markdown("""<div class="info-banner">Tras el análisis SHAP se identificaron <strong>VO (vacas en ordeñe)</strong> y <strong>kg LCGP (leche corregida por grasa y proteína)</strong> como las variables con mayor poder explicativo. Se verificó visualmente que la relación con las emisiones era lineal y se comparó un Random Forest con una Regresión Lineal usando sólo esas dos variables. El resultado fue que la regresión lineal superó al Random Forest.</div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Relación lineal entre variables y emisiones")
    st.markdown("""<div class="info-banner">Antes de elegir el modelo, se grafica cada variable candidata contra las emisiones totales para confirmar visualmente que la relación es aproximadamente lineal. Si los puntos siguen la línea de tendencia (roja), una regresión lineal es suficiente y mucho más interpretable que un modelo de ensamble.</div>""", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=len(VARS_MODELO_FINAL), subplot_titles=VARS_MODELO_FINAL)
    for i, var in enumerate(VARS_MODELO_FINAL, 1):
        df_plot = df_raw[[var, TARGET, 'ID', 'Provincia']].dropna()
        z = np.polyfit(df_plot[var], df_plot[TARGET], 1)
        p = np.poly1d(z)
        xvals = np.linspace(df_plot[var].min(), df_plot[var].max(), 100)
        fig.add_trace(go.Scatter(x=df_plot[var], y=df_plot[TARGET], mode='markers', marker=dict(color=COLORES['primary'], size=8, opacity=0.7, line=dict(color='white', width=1)), showlegend=False, customdata=df_plot[['ID', 'Provincia']].values, hovertemplate=f'<b>ID %{{customdata[0]}}</b><br>Prov: %{{customdata[1]}}<br>{var}: %{{x:,.1f}}<br>Emisiones: %{{y:,.0f}}<extra></extra>'), row=1, col=i)
        fig.add_trace(go.Scatter(x=xvals, y=p(xvals), mode='lines', line=dict(color='red', dash='dash', width=2), showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text=var, row=1, col=i)
        fig.update_yaxes(title_text='TNCO2eq/año', row=1, col=1)
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=420, margin=dict(l=20, r=20, t=60, b=40))
    fig.update_layout(title_text='Relación entre variables seleccionadas y emisiones totales')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚖️ Random Forest vs Regresión Lineal")
    st.markdown(f"""<div class="info-banner">A diferencia de los modelos iniciales (que usaban {n_vars_modelo} variables), aquí <strong>tanto el Random Forest como la Regresión Lineal se entrenan únicamente con VO y kg LCGP</strong>. Se comparan usando <strong>R² CV</strong> y <strong>RMSE</strong> (Root Mean Squared Error — error promedio en TNCO₂eq/año). Un RMSE más bajo significa que el modelo se equivoca menos en términos concretos.</div>""", unsafe_allow_html=True)

    rmse_rf = np.sqrt(mean_squared_error(m['y_te'], m['rf'].predict(m['X_te'])))
    rmse_lr = np.sqrt(mean_squared_error(m['y_te'], m['lr'].predict(m['X_te'])))
    cv_rmse_rf = np.sqrt(-cross_val_score(m['rf'], m['X_final'], m['y'], cv=5, scoring='neg_mean_squared_error'))
    cv_rmse_lr = np.sqrt(-cross_val_score(m['lr'], m['X_final'], m['y'], cv=5, scoring='neg_mean_squared_error'))

    col1, col2 = st.columns(2)
    for col, nombre, cv_r2, rmse_test, cv_rmse in [
        (col1, "🌲 Random Forest", m['cv_rf'], rmse_rf, cv_rmse_rf),
        (col2, "📐 Regresión Lineal", m['cv_lr'], rmse_lr, cv_rmse_lr),
    ]:
        col.markdown(f'<div class="metric-card"><div class="label">{nombre}</div><div class="value">{cv_r2.mean():.3f}</div><div class="delta">R² CV &nbsp;±&nbsp;{cv_r2.std():.3f}</div><div class="delta" style="margin-top:6px; color:#0284c7">RMSE CV: {cv_rmse.mean():,.1f} TNCO₂eq/año</div><div class="delta" style="color:#64748b">RMSE Test: {rmse_test:,.1f} TNCO₂eq/año</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='RMSE CV (5-fold)', x=['Random Forest', 'Regresión Lineal'], y=[cv_rmse_rf.mean(), cv_rmse_lr.mean()], error_y=dict(type='data', array=[cv_rmse_rf.std(), cv_rmse_lr.std()], visible=True), marker_color=COLORES['primary'], marker_line_color='white', marker_line_width=1, hovertemplate='<b>%{x}</b><br>RMSE CV: %{y:,.1f} TNCO₂eq/año<extra></extra>'))
    fig.add_trace(go.Bar(name='RMSE Test', x=['Random Forest', 'Regresión Lineal'], y=[rmse_rf, rmse_lr], marker_color=COLORES['light'], marker_line_color='white', marker_line_width=1, hovertemplate='<b>%{x}</b><br>RMSE Test: %{y:,.1f} TNCO₂eq/año<extra></extra>'))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], barmode='group', yaxis_title='RMSE (TNCO₂eq/año)', height=340, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="info-banner">💡 El RMSE expresa el error en las mismas unidades que las emisiones (TNCO₂eq/año), lo que permite interpretar concretamente cuánto se equivoca el modelo en promedio por tambo. Barras más bajas = mejor modelo.</div>""", unsafe_allow_html=True)

    # FIX 2: sección de ecuación completa restaurada
    st.markdown("---")
    st.markdown("### 🧮 Ecuación del Modelo")
    st.markdown("""<div class="info-banner">La fortaleza de la regresión lineal es que produce una <strong>ecuación explícita e interpretable</strong>. El intercepto representa las emisiones base independientes del tamaño. El coeficiente de VO indica cuántas TNCO₂eq adicionales genera cada vaca en ordeñe por año, y el de kg LCGP captura la contribución de la producción de leche.</div>""", unsafe_allow_html=True)
    coefs = dict(zip(VARS_MODELO_FINAL, m['lr'].coef_))
    intercept = m['lr'].intercept_
    eq = f"Emisiones = {intercept:.2f}"
    for var, coef in coefs.items():
        signo = '+' if coef >= 0 else '-'
        eq += f" {signo} {abs(coef):.4f} × {var}"
    st.code(eq, language=None)

    col1, col2, col3 = st.columns(3)
    items = [("Intercepto", f"{intercept:.2f}")] + [(v, f"{c:.4f}") for v, c in coefs.items()]
    for col, (var, val) in zip([col1, col2, col3], items):
        col.markdown(f'<div class="metric-card"><div class="label">{var}</div><div class="value" style="font-size:1.8rem">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Real vs Predicho — Modelo Final")
    st.markdown("""<div class="info-banner">La prueba definitiva: comparar las emisiones reales del set de test con las predichas por la regresión lineal. Un R² alto con puntos bien alineados sobre la diagonal confirma que el modelo generaliza bien, incluso con solo dos variables predictoras.</div>""", unsafe_allow_html=True)
    y_pred_lr = m['lr'].predict(m['X_te'])
    r2_lr = r2_score(m['y_te'], y_pred_lr)
    lims = [min(m['y_te'].min(), y_pred_lr.min()), max(m['y_te'].max(), y_pred_lr.max())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m['y_te'], y=y_pred_lr, mode='markers', marker=dict(color=COLORES['primary'], size=10, opacity=0.75, line=dict(color='white', width=1.5)), hovertemplate='Real: %{x:,.0f} TNCO2eq/año<br>Predicho: %{y:,.0f} TNCO2eq/año<extra></extra>', name='Tambos'))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', line=dict(color='red', dash='dash', width=2), name='Predicción perfecta'))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='Real (TNCO2eq/año)', yaxis_title='Predicho (TNCO2eq/año)', height=460, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=20, r=20, t=60, b=40))
    fig.update_layout(title=f'Regresión Lineal — R² Test = {r2_lr:.3f}')
    st.plotly_chart(fig, use_container_width=True)

# ════════ TAB 5 — CLUSTERS ════════
with tab5:
    st.markdown("## 🗂️ Perfiles de Tambos — Clustering")
    st.markdown("""<div class="info-banner">El análisis SHAP y el modelo final nos mostraron que <strong>VO y kg LCGP</strong> son las dos variables que mejor explican las emisiones. El clustering usa exactamente esas dos variables para agrupar los tambos en perfiles homogéneos. Una vez formados los grupos, se los <strong>caracteriza con todas las demás variables</strong> del dataset para entender en qué otras dimensiones se diferencian.</div>""", unsafe_allow_html=True)

    VARS_CLUSTER = ['VO', 'kg LCGP']
    vars_cluster_ok = [v for v in VARS_CLUSTER if v in df_raw.columns]
    VARS_CARACT = [v for v in ['Total leche kgCO2eq/lt', TARGET, 'Costo_USD', 'Costo_EUR', 'SUPERFICIE TOTAL TAMBO (Has)', 'CMS', '%PC', 'kgN total', 'Consumo Gasoil'] if v in df_raw.columns]
    cols_needed = list(dict.fromkeys([c for c in vars_cluster_ok + ['ID', 'Provincia', 'Localidad'] + VARS_CARACT if c in df_raw.columns]))
    df_cl = df_raw[cols_needed].dropna(subset=vars_cluster_ok).copy()
    X_cl = df_cl[vars_cluster_ok]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cl)

    st.markdown("### 📐 Selección del número de clusters")
    st.markdown("""<div class="info-banner">Se combinan dos criterios: el <strong>método del codo</strong> (donde la inercia deja de bajar significativamente) y el <strong>Silhouette Score</strong> (qué tan bien separados están los clusters, de -1 a 1, mayor es mejor). El slider arranca en el K sugerido por el Silhouette.</div>""", unsafe_allow_html=True)

    K_range = range(2, 9)
    inercias, silhouettes = [], []
    for k in K_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        inercias.append(km_tmp.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels_tmp))
    best_k = list(K_range)[silhouettes.index(max(silhouettes))]

    col_elbow, col_sil, col_slider = st.columns([5, 5, 3])
    with col_elbow:
        st.markdown("#### Método del Codo")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inercias, mode='lines+markers', line=dict(color=COLORES['primary'], width=2.5), marker=dict(color=COLORES['primary'], size=9, line=dict(color='white', width=2)), hovertemplate='K=%{x}<br>Inercia: %{y:,.0f}<extra></extra>'))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='K', yaxis_title='Inercia', height=280, margin=dict(l=20, r=20, t=10, b=40))
        fig.update_xaxes(tickmode='array', tickvals=list(K_range))
        st.plotly_chart(fig, use_container_width=True)

    with col_sil:
        st.markdown("#### Silhouette Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers', line=dict(color=COLORES['accent'], width=2.5), marker=dict(color=[COLORES['green'] if k == best_k else COLORES['accent'] for k in K_range], size=[14 if k == best_k else 9 for k in K_range], line=dict(color='white', width=2)), hovertemplate='K=%{x}<br>Silhouette: %{y:.3f}<extra></extra>'))
        fig.add_annotation(x=best_k, y=max(silhouettes), text=f'  Mejor K={best_k}', showarrow=False, font=dict(color=COLORES['green'], size=12, family='Outfit'), xanchor='left')
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='K', yaxis_title='Silhouette Score', height=280, margin=dict(l=20, r=20, t=10, b=40))
        fig.update_xaxes(tickmode='array', tickvals=list(K_range))
        st.plotly_chart(fig, use_container_width=True)

    with col_slider:
        st.markdown("#### Elegir K")
        n_clusters = st.slider("", min_value=2, max_value=8, value=best_k, step=1, label_visibility='collapsed')
        sil_val = silhouettes[n_clusters - 2]
        st.markdown(f'<div class="metric-card" style="margin-top:10px"><div class="label">Clusters seleccionados</div><div class="value">{n_clusters}</div><div class="delta">Silhouette: {sil_val:.3f}</div></div>', unsafe_allow_html=True)

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cl['Cluster'] = km_final.fit_predict(X_scaled)
    df_cl['Cluster_label'] = df_cl['Cluster'].apply(lambda x: f'Perfil {x+1}')

    def hex_rgba(hex_color, alpha=0.2):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    CLUSTER_COLORS = [COLORES['primary'], COLORES['accent'], COLORES['green'], '#a855f7', '#f43f5e', '#eab308', '#06b6d4', '#84cc16']

    st.markdown("---")
    st.markdown("### 🔵 Clusters en el espacio VO — kg LCGP")
    st.markdown("""<div class="info-banner">Los clusters se forman en el espacio de las dos variables del modelo final. Cada punto es un tambo; el color indica el perfil asignado. Las estrellas muestran los <strong>centroides</strong> de cada cluster.</div>""", unsafe_allow_html=True)

    fig = go.Figure()
    for i in range(n_clusters):
        mask = df_cl['Cluster'] == i
        df_c = df_cl[mask]
        c = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        hover_cols = ['ID', 'Provincia', TARGET, 'Costo_USD'] if TARGET in df_c.columns else ['ID', 'Provincia']
        fig.add_trace(go.Scatter(x=df_c['VO'], y=df_c['kg LCGP'], mode='markers', name=f'Perfil {i+1}', marker=dict(color=c, size=9, opacity=0.8, line=dict(color='white', width=1.5)), customdata=df_c[hover_cols].values, hovertemplate=f'<b>Perfil {i+1} — ID %{{customdata[0]}}</b><br>Provincia: %{{customdata[1]}}<br>VO: %{{x:,.0f}} vacas<br>kg LCGP: %{{y:,.0f}} kg/año<br>Emisiones: %{{customdata[2]:,.0f}} TNCO₂eq/año<br>Costo: $%{{customdata[3]:,.0f}} USD<extra></extra>'))
        cx, cy = df_c['VO'].mean(), df_c['kg LCGP'].mean()
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers+text', marker=dict(color=c, size=18, symbol='star', line=dict(color='white', width=2)), text=[f'P{i+1}'], textposition='top center', textfont=dict(size=11, color=c, family='Outfit'), showlegend=False, hovertemplate=f'<b>Centroide Perfil {i+1}</b><br>VO medio: {cx:,.0f}<br>kg LCGP medio: {cy:,.0f}<extra></extra>'))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='Vacas en Ordeñe (VO)', yaxis_title='kg LCGP (leche corregida/año)', height=480, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Costo Ambiental por Perfil")
    cols_cost = st.columns(n_clusters)
    for i, col in enumerate(cols_cost):
        mask = df_cl['Cluster'] == i
        n = mask.sum()
        emis_med = df_cl.loc[mask, TARGET].mean() if TARGET in df_cl.columns else 0
        costo_med = df_cl.loc[mask, 'Costo_USD'].mean()
        col.markdown(f'<div class="metric-card"><div class="label">Perfil {i+1} · {n} tambos</div><div class="value" style="font-size:1.6rem">${costo_med:,.0f}</div><div class="delta">USD/año · {emis_med:,.0f} TNCO₂eq</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Caracterización de Perfiles")
    st.markdown("""<div class="info-banner">Los clusters se formaron usando solo <strong>VO y kg LCGP</strong>. Esta tabla muestra cómo se diferencian los perfiles en las <strong>demás variables del dataset</strong>, que no participaron en la agrupación.</div>""", unsafe_allow_html=True)
    vars_tabla = [v for v in ['VO', 'kg LCGP', TARGET, 'Costo_USD', 'Total leche kgCO2eq/lt', 'SUPERFICIE TOTAL TAMBO (Has)', 'CMS', '%PC', 'kgN total', 'Consumo Gasoil'] if v in df_cl.columns]
    perfil_df = df_cl.groupby('Cluster_label')[vars_tabla].mean().round(2).T
    rename_map = {TARGET: 'Emisiones (TNCO₂eq/año)', 'Costo_USD': 'Costo ambiental (USD/año)', 'Total leche kgCO2eq/lt': 'kgCO₂eq por litro', 'SUPERFICIE TOTAL TAMBO (Has)': 'Superficie (has)', 'kgN total': 'Fertilización N (kg)', 'Consumo Gasoil': 'Gasoil (l/año)'}
    perfil_df.index = [rename_map.get(i, i) for i in perfil_df.index]
    st.markdown("**🔵 Variables usadas para clusterizar** (primeras 2 filas) · **Variables de caracterización** (resto)")
    st.dataframe(perfil_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📦 Distribución de Emisiones y Costo por Perfil")
    st.markdown("""<div class="info-banner">Los box plots muestran la dispersión <em>dentro</em> de cada perfil. La línea central es la mediana, la caja los cuartiles 25-75% y los bigotes el rango. La cruz (×) indica la media. Un perfil con caja angosta es homogéneo; bigotes largos indican tambos muy distintos entre sí dentro del grupo.</div>""", unsafe_allow_html=True)
    col_box1, col_box2 = st.columns(2)
    with col_box1:
        st.markdown("#### Emisiones Totales (TNCO₂eq/año)")
        fig = go.Figure()
        for i in range(n_clusters):
            vals = df_cl[df_cl['Cluster'] == i][TARGET] if TARGET in df_cl.columns else pd.Series()
            c = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            fig.add_trace(go.Box(y=vals, name=f'Perfil {i+1}', marker_color=c, line_color=c, fillcolor=hex_rgba(c, 0.2), boxmean=True, hovertemplate='%{y:,.0f} TNCO₂eq/año<extra></extra>'))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], yaxis_title='TNCO₂eq/año', showlegend=False, height=360, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with col_box2:
        st.markdown("#### Costo Ambiental (USD/año)")
        fig = go.Figure()
        for i in range(n_clusters):
            vals = df_cl[df_cl['Cluster'] == i]['Costo_USD']
            c = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            fig.add_trace(go.Box(y=vals, name=f'Perfil {i+1}', marker_color=c, line_color=c, fillcolor=hex_rgba(c, 0.2), boxmean=True, hovertemplate='$%{y:,.0f} USD/año<extra></extra>'))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], yaxis_title='USD/año', showlegend=False, height=360, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗺️ Distribución de Perfiles por Provincia")
    st.markdown("""<div class="info-banner">¿Hay provincias donde predomina algún perfil en particular? Esto permitiría identificar si los sistemas productivos tienen patrones geográficos sistemáticos.</div>""", unsafe_allow_html=True)
    prov_cluster = df_cl.groupby(['Provincia', 'Cluster_label']).size().reset_index(name='n')
    fig = go.Figure()
    for i in range(n_clusters):
        label = f'Perfil {i+1}'
        datos = prov_cluster[prov_cluster['Cluster_label'] == label]
        fig.add_trace(go.Bar(name=label, x=datos['Provincia'], y=datos['n'], marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], hovertemplate=f'<b>{label}</b><br>Provincia: %{{x}}<br>Tambos: %{{y}}<extra></extra>'))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], barmode='stack', xaxis_title='Provincia', yaxis_title='Cantidad de tambos', height=360, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ════════ TAB 6 — PREDICTOR ════════
with tab6:
    st.markdown("## Predictor de Emisiones y Costo Ambiental")
    st.markdown(f"""<div class="info-banner">Ingresar los datos de un tambo para predecir sus emisiones anuales y el costo ambiental equivalente bajo el esquema de impuesto al carbono danés (€{PRECIO_CARBONO_EUR}/tCO₂eq → ${PRECIO_CARBONO_USD:.2f} USD/tCO₂eq).</div>""", unsafe_allow_html=True)

    st.markdown("### Ingresá los datos del tambo")
    col1, col2 = st.columns(2)
    with col1:
        vo_input = st.number_input("🐄 Vacas en Ordeñe (VO)", min_value=0, max_value=2000, value=int(df_raw['VO'].mean()), step=1)
    with col2:
        lcgp_input = st.number_input("🥛 kg LCGP (kg leche corregida/año)", min_value=0, max_value=10000000, value=int(df_raw['kg LCGP'].mean()), step=1000)

    st.markdown("---")
    if st.button("🔍 Calcular Emisiones y Costo Ambiental"):
        datos = pd.DataFrame([[vo_input, lcgp_input]], columns=VARS_MODELO_FINAL)
        emisiones = m['lr'].predict(datos)[0]
        costo_usd = emisiones * PRECIO_CARBONO_USD
        costo_eur = emisiones * PRECIO_CARBONO_EUR

        st.markdown("### Resultado")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="result-box"><div class="rlabel">Emisiones estimadas</div><div class="big-number">{emisiones:,.1f}</div><div class="sub">TNCO₂eq / año</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="result-box"><div class="rlabel">Costo ambiental (USD)</div><div class="big-number">${costo_usd:,.0f}</div><div class="sub">USD / año · ${PRECIO_CARBONO_USD:.2f} por tCO₂eq</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="result-box"><div class="rlabel">Costo ambiental (EUR)</div><div class="big-number">€{costo_eur:,.0f}</div><div class="sub">EUR / año · €{PRECIO_CARBONO_EUR} por tCO₂eq</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ¿Cómo se ubica este tambo en el dataset?")
        emisiones_dataset = m['lr'].predict(m['X_final'])
        percentil = (emisiones_dataset < emisiones).mean() * 100

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=emisiones_dataset, nbinsx=20, marker_color=COLORES['primary'], marker_line_color='white', marker_line_width=1, opacity=0.7, name='Tambos del dataset', hovertemplate='Rango: %{x}<br>Frecuencia: %{y}<extra></extra>'))
        fig.add_vline(x=emisiones, line_color='#1a1a1a', line_width=2.5, line_dash='dash', annotation_text=f"Este tambo: {emisiones:,.1f}", annotation_position='top right', annotation_font_color='#1a1a1a', annotation_font_size=13)
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], xaxis_title='Emisiones predichas (TNCO₂eq/año)', yaxis_title='Frecuencia', height=380, margin=dict(l=20, r=20, t=20, b=40), legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        if percentil < 33:
            categoria, color_cat = "✅ Bajo emisor", "#2d7a27"
        elif percentil < 66:
            categoria, color_cat = "⚠️ Emisor medio", "#c47800"
        else:
            categoria, color_cat = "🔴 Alto emisor", "#c40000"

        st.markdown(f'<div class="info-banner">Este tambo emite más que el <strong>{percentil:.0f}%</strong> de los tambos del dataset. Categoría: <strong style="color:{color_cat}">{categoria}</strong></div>', unsafe_allow_html=True)
