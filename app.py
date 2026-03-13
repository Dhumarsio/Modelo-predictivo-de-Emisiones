import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Cluster de tambos para proyecto final",
    page_icon="🐄",
    layout="wide"
)

TASA          = 16
K_OPTIMO      = 3
ARCHIVO_NUEVOS = "tambos_nuevos.csv"

NOMBRES_CLUSTER = {
    0: "✅ Tambos eficientes",
    1: "🔴 Tambos con Alto riesgo",
    2: "🟢 Tambos intermedios",
}
COLORES_CLUSTER = {
    0: "#16a34a",
    1: "#dc2626",
    2: "#3b82f6",
}

features_cluster = [
    "sup_total_ha",
    "carga_vt",
    "vt",
    "lcg",
    "cms",
    "lts_dia",
    "total_estab_tnco2",
    "huella_kgco2_lt",
]


# CARGA Y PROCESAMIENTO
@st.cache_data
def cargar_y_procesar():
    df_raw = pd.read_csv("datostambo.csv", sep=";", header=0)

    categoricas = ["Provincia", "Localidad", "CUENCA LECHERA", "Sistema alimentación"]
    for col in df_raw.columns:
        if col not in categoricas:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw = df_raw.rename(columns={
        "ID. Productor:":                                "id",
        "CUENCA LECHERA":                                "cuenca",
        "Sistema alimentación":                          "sistema",
        "SUPERFICIE TOTAL TAMBO (Has)":                  "sup_total_ha",
        "VO":                                            "vo",
        "VS":                                            "vs",
        "VT":                                            "vt",
        "Carga VT":                                      "carga_vt",
        "Kg MS concentrado / VO / dia - promedio anual": "concentrado",
        "CMS":                                           "cms",
        "CMS pastura":                                   "cms_pastura",
        "lts":                                           "lts_dia",
        "%GB":                                           "pct_gb",
        "%PC":                                           "pct_pc",
        "LCG":                                           "lcg",
        "Terneras recría (< 12 meses)":                  "terneras_recria",
        "Total Tambo TNCO2eq / año":                     "total_tambo_tnco2",
        "Total establecimiento TNCO2eq / año":           "total_estab_tnco2",
        "Total leche KgCO2eq / Lt":                      "huella_kgco2_lt",
    })

    df_raw["pct_gb"] = df_raw["pct_gb"].replace(0, np.nan)
    df = df_raw.dropna(subset=["huella_kgco2_lt"]).copy()
    df = df[df["id"] != 102].copy()
    df["costo_ambiental_eur"] = df["total_estab_tnco2"] * TASA
    return df


@st.cache_data
def entrenar_clustering(df):
    df_c = df[["id"] + features_cluster].dropna().copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_c[features_cluster])

    km = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
    df_c["cluster"] = km.fit_predict(X_scaled)
    df_c["perfil"]  = df_c["cluster"].map(NOMBRES_CLUSTER)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_c["PC1"] = coords[:, 0]
    df_c["PC2"] = coords[:, 1]

    df_resultado = df.merge(df_c[["id", "cluster", "perfil", "PC1", "PC2"]],
                            on="id", how="inner")
    return df_resultado, scaler, km, pca


def cargar_nuevos():
    if os.path.exists(ARCHIVO_NUEVOS):
        return pd.read_csv(ARCHIVO_NUEVOS)
    return pd.DataFrame()


def guardar_nuevo(fila):
    df_existentes = cargar_nuevos()
    df_total = pd.concat([df_existentes, pd.DataFrame([fila])], ignore_index=True)
    df_total.to_csv(ARCHIVO_NUEVOS, index=False)


def clasificar_nuevo(datos, scaler, km, pca):
    X = pd.DataFrame([datos])[features_cluster]
    X_scaled = scaler.transform(X)
    cluster  = km.predict(X_scaled)[0]
    coords   = pca.transform(X_scaled)[0]
    return cluster, coords


# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────
df = cargar_y_procesar()
df, scaler, km, pca = entrenar_clustering(df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🐄 Cluster para proyecto final")
st.caption(f"Escenario regulatorio basado en tasa danesa: EUR {TASA} / tn CO2eq")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros")
    tasa_limite = st.slider(
        "Tasa límite (KgCO2eq / litro)",
        min_value=0.60, max_value=1.50,
        value=1.00, step=0.05,
        help="Umbral regulatorio. Tambos que superen este valor están en riesgo."
    )
    st.divider()
    st.caption(f"Tasa de cobro: EUR {TASA} / tn CO2eq")

# ─────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen informativo",
    "🔵 Segmentación de tambos",
    "🎚️ Simulador de tasa",
    "➕ Agregar tambo nuevo",
])

# ════════════════════════════════════════════
# TAB 1 — RESUMEN EJECUTIVO
# ════════════════════════════════════════════
with tab1:
    st.subheader("Resumen informativo")

    en_riesgo    = df[df["huella_kgco2_lt"] > tasa_limite]
    costo_total  = df["costo_ambiental_eur"].sum()
    costo_riesgo = en_riesgo["costo_ambiental_eur"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tambos", f"{len(df)}")
    col2.metric("Tambos en riesgo", f"{len(en_riesgo)}",
                delta=f"{len(en_riesgo)/len(df)*100:.1f}% del total",
                delta_color="inverse")
    col3.metric("Costo total tambos",
                f"EUR {costo_total:,.0f}",
                help="Costo ambiental anual de toda los tambos")
    col4.metric("Costo en riesgo",
                f"EUR {costo_riesgo:,.0f}",
                delta=f"{costo_riesgo/costo_total*100:.1f}% del total",
                delta_color="inverse")

    st.divider()

    # Costo por cluster
    st.subheader("Costo ambiental por perfil")
    resumen = df.groupby("perfil").agg(
        Tambos          = ("id", "count"),
        Huella_prom     = ("huella_kgco2_lt", "mean"),
        Huella_max      = ("huella_kgco2_lt", "max"),
        Costo_prom_EUR  = ("costo_ambiental_eur", "mean"),
        Costo_total_EUR = ("costo_ambiental_eur", "sum"),
        Lts_prom        = ("lts_dia", "mean"),
    ).round(2).reset_index()
    resumen.columns = ["Perfil", "Tambos", "Huella prom.",
                       "Huella máx.", "Costo prom. EUR",
                       "Costo total EUR", "Lts/día prom."]
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.divider()

    # Costo por tambo
    st.subheader("Costo ambiental por tambo")
    df_ord = df.sort_values("costo_ambiental_eur", ascending=False).copy()
    df_ord["estado"] = df_ord["huella_kgco2_lt"].apply(
        lambda x: "Supera tasa" if x > tasa_limite else "Cumple tasa"
    )

    fig_costos = go.Figure()
    for estado, color in [("Cumple tasa", "#3b82f6"), ("Supera tasa", "#dc2626")]:
        mask = df_ord["estado"] == estado
        fig_costos.add_trace(go.Bar(
            x=df_ord.loc[mask, "id"].astype(str),
            y=df_ord.loc[mask, "costo_ambiental_eur"],
            name=estado,
            marker_color=color,
            hovertemplate=(
                "<b>Tambo %{x}</b><br>"
                "Costo: EUR %{y:,.0f}<br>"
                "<extra></extra>"
            )
        ))
    fig_costos.update_layout(
        title="Costo ambiental por tambo (EUR/año)",
        xaxis_title="ID Tambo",
        yaxis_title="EUR / año",
        barmode="overlay",
        height=400,
        yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_costos, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — SEGMENTACIÓN
# ════════════════════════════════════════════
with tab2:
    st.subheader("Segmentación de tambos por perfil")

    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        fig_pca = px.scatter(
            df, x="PC1", y="PC2",
            color="perfil",
            color_discrete_map={v: COLORES_CLUSTER[k]
                                 for k, v in NOMBRES_CLUSTER.items()},
            hover_data=["id", "huella_kgco2_lt", "costo_ambiental_eur",
                        "vt", "lts_dia", "sistema", "cuenca"],
            labels={
                "PC1": "Tamaño del establecimiento",
                "PC2": "Eficiencia productiva",
                "perfil": "Perfil",
            },
            title="Mapa de clusters (PCA)",
        )
        fig_pca.update_layout(height=450)
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_der:
        st.markdown("**Distribución por sistema de alimentación**")
        fig_sist = px.histogram(
            df, x="sistema", color="perfil",
            color_discrete_map={v: COLORES_CLUSTER[k]
                                 for k, v in NOMBRES_CLUSTER.items()},
            barmode="group",
            labels={"sistema": "Sistema", "perfil": "Perfil"},
        )
        fig_sist.update_layout(height=220, showlegend=False)
        st.plotly_chart(fig_sist, use_container_width=True)

        st.markdown("**Distribución por cuenca**")
        fig_cuenca = px.histogram(
            df, x="cuenca", color="perfil",
            color_discrete_map={v: COLORES_CLUSTER[k]
                                 for k, v in NOMBRES_CLUSTER.items()},
            barmode="group",
            labels={"cuenca": "Cuenca", "perfil": "Perfil"},
        )
        fig_cuenca.update_layout(height=220, showlegend=False,
                                  xaxis_tickangle=-30)
        st.plotly_chart(fig_cuenca, use_container_width=True)

    st.divider()

    # Tabla filtrable
    st.subheader("Composición de emisiones por cluster")

    fuentes = {
        "Fermentación enterica tambo CH4 TNCO2eq / año": "pct_ferment_enteric",
        "Fermentacion entérica Recría TNCO2eq / año":    "pct_ferment_recria",
        "Animales CH4 TNCO2eq / año":                    "pct_animales_ch4",
        "Animales N2O TNCO2eq / año":                    "pct_animales_n2o",
        "Fertilizante N2O TNCO2eq / año":                "pct_fertilizante",
        "Fertilizante (propio) N2O TNCO2eq / año":       "pct_fertilizante_propio",
        "Energia CO2 TNCO2eq / año":                     "pct_energia",
        "Manejo Estiercol CH4 TNCO2eq / año":            "pct_manejo_estiercol",
        "Estiercol en pasturas N2O TNCO2eq / año":       "pct_estiercol_n2o",
    }

    for col_orig, col_nuevo in fuentes.items():
        df[col_nuevo] = df[col_orig] / df["total_estab_tnco2"]

    cols_pct = list(fuentes.values())
    etiquetas = [
        "Ferment. entérica tambo",
        "Ferment. entérica recría",
        "Animales CH4",
        "Animales N2O",
        "Fertilizante",
        "Fertilizante propio",
        "Energía",
        "Manejo estiércol",
        "Estiércol pasturas",
    ]

    df_comp = df.groupby("perfil")[cols_pct].mean().reset_index()
    df_comp.columns = ["Perfil"] + etiquetas

    fig_comp = go.Figure()
    colores_fuente = [
        "#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd",
        "#16a34a", "#4ade80",
        "#dc2626",
        "#f97316", "#fb923c",
    ]

    for etiqueta, color in zip(etiquetas, colores_fuente):
        fig_comp.add_trace(go.Bar(
            name=etiqueta,
            x=df_comp["Perfil"],
            y=df_comp[etiqueta],
            marker_color=color,
        ))

    fig_comp.update_layout(
        barmode="stack",
        barnorm="percent",
        title="Composición promedio de emisiones por perfil",
        yaxis_title="Proporción sobre total",
        height=450,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Tabla de tambos")
    perfil_filtro = st.multiselect(
        "Filtrar por perfil",
        options=list(NOMBRES_CLUSTER.values()),
        default=list(NOMBRES_CLUSTER.values())
    )
    df_filtrado = df[df["perfil"].isin(perfil_filtro)][[
        "id", "cuenca", "sistema", "perfil", "vt", "lts_dia",
        "huella_kgco2_lt", "total_estab_tnco2", "costo_ambiental_eur"
    ]].rename(columns={
        "id":                  "ID",
        "cuenca":              "Cuenca",
        "sistema":             "Sistema",
        "perfil":              "Perfil",
        "vt":                  "VT",
        "lts_dia":             "Lts/día",
        "huella_kgco2_lt":     "Huella KgCO2/lt",
        "total_estab_tnco2":   "Emisiones tn CO2",
        "costo_ambiental_eur": "Costo EUR/año",
    })
    st.dataframe(df_filtrado.round(3), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 3 — SIMULADOR
# ════════════════════════════════════════════
with tab3:
    st.subheader("Simulador de tasa ambiental")
    st.caption("Mové el slider en el panel izquierdo para cambiar el umbral regulatorio")

    col_a, col_b = st.columns(2)

    with col_a:
        df["estado"] = df["huella_kgco2_lt"].apply(
            lambda x: "Supera tasa" if x > tasa_limite else "Cumple tasa"
        )
        fig_sim = px.scatter(
            df, x="huella_kgco2_lt", y="costo_ambiental_eur",
            color="estado",
            color_discrete_map={
                "Cumple tasa":  "#16a34a",
                "Supera tasa":  "#dc2626"
            },
            hover_data=["id", "cuenca", "sistema", "vt"],
            labels={
                "huella_kgco2_lt":     "Huella (KgCO2eq/lt)",
                "costo_ambiental_eur": "Costo ambiental (EUR/año)",
                "estado":              "Estado",
            },
            title="Huella vs Costo ambiental",
        )
        fig_sim.add_vline(x=tasa_limite, line_dash="dash", line_color="red",
                          annotation_text=f"Límite: {tasa_limite}")
        fig_sim.update_layout(height=400)
        st.plotly_chart(fig_sim, use_container_width=True)

    with col_b:
        en_riesgo_sim = df[df["huella_kgco2_lt"] > tasa_limite]
        cumplen_sim   = df[df["huella_kgco2_lt"] <= tasa_limite]

        st.metric("Tambos que superan la tasa",
                  f"{len(en_riesgo_sim)}",
                  delta=f"{len(en_riesgo_sim)/len(df)*100:.1f}% del total",
                  delta_color="inverse")
        st.metric("Tambos que cumplen",
                  f"{len(cumplen_sim)}")
        st.metric("Costo en riesgo",
                  f"EUR {en_riesgo_sim['costo_ambiental_eur'].sum():,.0f}",
                  delta_color="inverse")

        st.divider()
        st.markdown("**Tambos en riesgo por perfil**")
        riesgo_perfil = (en_riesgo_sim.groupby("perfil")["id"]
                         .count().reset_index()
                         .rename(columns={"id": "Tambos en riesgo", "perfil": "Perfil"}))
        st.dataframe(riesgo_perfil, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 4 — AGREGAR TAMBO NUEVO
# ════════════════════════════════════════════
with tab4:
    st.subheader("Clasificar tambo nuevo")
    st.caption("Ingresá los datos del tambo y el modelo lo asignará al cluster correspondiente")

    with st.form("form_nuevo"):
        col1, col2, col3 = st.columns(3)

        with col1:
            nombre_tambo     = st.text_input("Nombre o ID del tambo", value="Tambo nuevo")
            sup_total_ha     = st.number_input("Superficie total (ha)",
                                               min_value=50.0, max_value=5000.0,
                                               value=300.0, step=10.0)
            carga_vt         = st.number_input("Carga VT (VT/ha)",
                                               min_value=0.5, max_value=5.0,
                                               value=1.4, step=0.1)
            vt               = st.number_input("VT total (cabezas)",
                                               min_value=20, max_value=3000,
                                               value=200)

        with col2:
            lcg              = st.number_input("LCG (litros corregidos grasa)",
                                               min_value=5.0, max_value=40.0,
                                               value=18.0, step=0.5)
            cms              = st.number_input("CMS (kg MS/día)",
                                               min_value=5.0, max_value=35.0,
                                               value=18.0, step=0.5)
            lts_dia          = st.number_input("Litros / vaca / día",
                                               min_value=5.0, max_value=50.0,
                                               value=18.0, step=0.5)

        with col3:
            total_estab_tnco2 = st.number_input("Emisiones totales (tn CO2eq/año)",
                                                min_value=50.0, max_value=5000.0,
                                                value=800.0, step=10.0)
            huella_kgco2_lt   = st.number_input("Huella (KgCO2eq/lt)",
                                                min_value=0.3, max_value=2.5,
                                                value=0.85, step=0.01)
            cuenca            = st.selectbox("Cuenca",
                                             options=sorted(df["cuenca"].unique()))
            sistema           = st.selectbox("Sistema de alimentación",
                                             options=["PMR", "TMR", "sep"])

        submitted = st.form_submit_button("🔍 Clasificar tambo",
                                          use_container_width=True)

    if submitted:
        datos_nuevo = {
            "sup_total_ha":      sup_total_ha,
            "carga_vt":          carga_vt,
            "vt":                vt,
            "lcg":               lcg,
            "cms":               cms,
            "lts_dia":           lts_dia,
            "total_estab_tnco2": total_estab_tnco2,
            "huella_kgco2_lt":   huella_kgco2_lt,
        }

        cluster_nuevo, coords_nuevo = clasificar_nuevo(datos_nuevo, scaler, km, pca)
        perfil_nuevo  = NOMBRES_CLUSTER[cluster_nuevo]
        color_nuevo   = COLORES_CLUSTER[cluster_nuevo]
        costo_nuevo   = total_estab_tnco2 * TASA

        st.divider()
        st.markdown(f"### Resultado: {perfil_nuevo}")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Perfil asignado",     perfil_nuevo)
        col_r2.metric("Costo ambiental",     f"EUR {costo_nuevo:,.0f} / año")
        col_r3.metric("Huella",              f"{huella_kgco2_lt:.3f} KgCO2eq/lt",
                      delta="⚠️ Supera tasa" if huella_kgco2_lt > tasa_limite else "✅ Cumple tasa",
                      delta_color="inverse" if huella_kgco2_lt > tasa_limite else "normal")

        # Mostrar en el mapa PCA
        fig_nuevo = px.scatter(
            df, x="PC1", y="PC2", color="perfil",
            color_discrete_map={v: COLORES_CLUSTER[k]
                                 for k, v in NOMBRES_CLUSTER.items()},
            opacity=0.4,
            title=f"Posición de '{nombre_tambo}' en el mapa de clusters",
        )
        fig_nuevo.add_scatter(
            x=[coords_nuevo[0]], y=[coords_nuevo[1]],
            mode="markers+text",
            marker=dict(size=18, color=color_nuevo, symbol="star",
                        line=dict(color="white", width=2)),
            text=[nombre_tambo],
            textposition="top center",
            name=nombre_tambo,
        )
        fig_nuevo.update_layout(height=400)
        st.plotly_chart(fig_nuevo, use_container_width=True)

        # Guardar
        guardar_nuevo({
            "nombre":            nombre_tambo,
            "cuenca":            cuenca,
            "sistema":           sistema,
            "sup_total_ha":      sup_total_ha,
            "carga_vt":          carga_vt,
            "vt":                vt,
            "lcg":               lcg,
            "cms":               cms,
            "lts_dia":           lts_dia,
            "total_estab_tnco2": total_estab_tnco2,
            "huella_kgco2_lt":   huella_kgco2_lt,
            "costo_eur":         costo_nuevo,
            "perfil":            perfil_nuevo,
        })
        st.success(f"✅ Tambo '{nombre_tambo}' guardado correctamente.")

    # Tambos nuevos guardados
    st.divider()
    st.subheader("Tambos nuevos registrados")
    df_nuevos = cargar_nuevos()

    if df_nuevos.empty:
        st.info("Todavía no hay tambos nuevos registrados.")
    else:
        st.dataframe(df_nuevos, use_container_width=True, hide_index=True)
        col_n1, col_n2 = st.columns(2)
        col_n1.metric("Total registrados", len(df_nuevos))
        col_n2.metric("Costo total nuevos",
                      f"EUR {df_nuevos['costo_eur'].sum():,.0f}")

        if st.button("🗑️ Borrar todos los tambos nuevos"):
            os.remove(ARCHIVO_NUEVOS)
            st.rerun()
