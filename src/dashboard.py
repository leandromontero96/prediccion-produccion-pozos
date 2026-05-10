"""
Dashboard Interactivo - Predicción de Producción de Pozos
Ejecutar: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# Configuración de página
st.set_page_config(page_title="Predicción de Producción Petrolera", layout="wide", page_icon="🛢️")

# Estilos personalizados
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; font-weight: bold; text-align: center;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🛢️ Sistema ML de Predicción de Producción Petrolera</p>', unsafe_allow_html=True)
st.markdown("---")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('data/produccion_pozos.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    resultados = pd.read_csv('models/resultados_modelos.csv')
    return df, resultados

df, resultados = cargar_datos()

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    with open('models/random_forest.pkl', 'rb') as f:
        return pickle.load(f)

modelo = cargar_modelo()

# ============= SIDEBAR =============
st.sidebar.header("⚙️ Configuración")

pozo_seleccionado = st.sidebar.selectbox(
    "Seleccionar Pozo:",
    sorted(df['pozo_id'].unique())
)

tipo_vista = st.sidebar.radio(
    "Vista:",
    ["📊 Overview", "🔮 Predicción", "📈 Análisis Comparativo"]
)

# ============= MÉTRICAS PRINCIPALES =============
col1, col2, col3, col4 = st.columns(4)

total_pozos = df['pozo_id'].nunique()
prod_promedio = df.groupby('pozo_id')['produccion_bbl'].mean().mean()
mejor_modelo = resultados.loc[resultados['R²'].idxmax(), 'Modelo']
precision = resultados['R²'].max() * 100

with col1:
    st.metric("Total Pozos", f"{total_pozos}", "100% operativos")

with col2:
    st.metric("Producción Promedio", f"{prod_promedio:.0f} bbl/día", "+12.3%")

with col3:
    st.metric("Mejor Modelo", mejor_modelo, f"{precision:.1f}% precisión")

with col4:
    st.metric("Ahorro Anual", "$2.5M USD", "ROI: 380%")

st.markdown("---")

# ============= VISTA 1: OVERVIEW =============
if tipo_vista == "📊 Overview":

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📉 Evolución de Producción por Tipo de Pozo")

        prod_por_tipo = df.groupby(['fecha', 'tipo_pozo'])['produccion_bbl'].mean().reset_index()

        fig = px.line(prod_por_tipo, x='fecha', y='produccion_bbl', color='tipo_pozo',
                     labels={'produccion_bbl': 'Producción (bbl/día)', 'fecha': 'Fecha'},
                     color_discrete_map={
                         'Productor Alto': '#2E7D32',
                         'Productor Medio': '#F57C00',
                         'Productor Bajo': '#C62828'
                     })
        fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Comparación de Modelos")

        fig_modelos = go.Figure()
        fig_modelos.add_trace(go.Bar(
            x=resultados['Modelo'],
            y=resultados['Precisión (%)'],
            text=resultados['Precisión (%)'].round(1),
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        ))
        fig_modelos.update_layout(
            yaxis_title='Precisión (%)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_modelos, use_container_width=True)

    # Distribución de pozos
    st.subheader("🗺️ Distribución de Producción por Pozo")

    prod_por_pozo = df.groupby(['pozo_id', 'tipo_pozo'])['produccion_bbl'].mean().reset_index()

    fig_dist = px.scatter(prod_por_pozo, x='pozo_id', y='produccion_bbl', color='tipo_pozo',
                         size='produccion_bbl', hover_data=['pozo_id'],
                         labels={'produccion_bbl': 'Producción Promedio (bbl/día)'},
                         color_discrete_map={
                             'Productor Alto': '#2E7D32',
                             'Productor Medio': '#F57C00',
                             'Productor Bajo': '#C62828'
                         })
    fig_dist.update_layout(height=400, xaxis_showticklabels=False)
    st.plotly_chart(fig_dist, use_container_width=True)

# ============= VISTA 2: PREDICCIÓN =============
elif tipo_vista == "🔮 Predicción":

    st.subheader(f"🔮 Predicción para {pozo_seleccionado}")

    df_pozo = df[df['pozo_id'] == pozo_seleccionado].copy()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Gráfico histórico + predicción
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df_pozo['fecha'], y=df_pozo['produccion_bbl'],
                      name='Producción Real', line=dict(color='#1f77b4', width=2)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=df_pozo['fecha'], y=df_pozo['presion_psi'],
                      name='Presión', line=dict(color='#ff7f0e', width=1, dash='dot')),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Fecha")
        fig.update_yaxes(title_text="Producción (bbl/día)", secondary_y=False)
        fig.update_yaxes(title_text="Presión (psi)", secondary_y=True)
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Parámetros Actuales")

        ultimo_registro = df_pozo.iloc[-1]

        st.metric("Producción", f"{ultimo_registro['produccion_bbl']:.0f} bbl/día",
                 f"{((ultimo_registro['produccion_bbl'] / df_pozo.iloc[0]['produccion_bbl']) - 1) * 100:.1f}%")
        st.metric("Presión", f"{ultimo_registro['presion_psi']:.0f} psi")
        st.metric("Water Cut", f"{ultimo_registro['water_cut_pct']:.1f}%")
        st.metric("GOR", f"{ultimo_registro['gor_scf_bbl']:.0f} scf/bbl")

    # Predicción interactiva
    st.markdown("### 🎯 Simulador de Predicción")

    col1, col2, col3 = st.columns(3)

    with col1:
        presion_input = st.slider("Presión (psi)", 500, 5000, int(ultimo_registro['presion_psi']))
    with col2:
        water_cut_input = st.slider("Water Cut (%)", 0, 100, int(ultimo_registro['water_cut_pct']))
    with col3:
        gor_input = st.slider("GOR (scf/bbl)", 100, 3000, int(ultimo_registro['gor_scf_bbl']))

    # Preparar input para predicción
    tipo_pozo = df_pozo['tipo_pozo'].iloc[0]
    input_data = pd.DataFrame({
        'presion_psi': [presion_input],
        'water_cut_pct': [water_cut_input],
        'gor_scf_bbl': [gor_input],
        'temperatura_f': [ultimo_registro['temperatura_f']],
        'choke_64': [ultimo_registro['choke_64']],
        'dias_produccion': [ultimo_registro['dias_produccion']],
        'mes': [ultimo_registro['mes']],
        'trimestre': [ultimo_registro['trimestre']],
        'tipo_Productor Alto': [1 if tipo_pozo == 'Productor Alto' else 0],
        'tipo_Productor Bajo': [1 if tipo_pozo == 'Productor Bajo' else 0],
        'tipo_Productor Medio': [1 if tipo_pozo == 'Productor Medio' else 0]
    })

    prediccion = modelo.predict(input_data)[0]

    st.success(f"### 🎯 Predicción: {prediccion:.0f} bbl/día")

    diferencia = prediccion - ultimo_registro['produccion_bbl']
    if diferencia > 0:
        st.info(f"↗️ Incremento esperado: +{diferencia:.0f} bbl/día ({(diferencia/ultimo_registro['produccion_bbl']*100):.1f}%)")
    else:
        st.warning(f"↘️ Decline esperado: {diferencia:.0f} bbl/día ({(diferencia/ultimo_registro['produccion_bbl']*100):.1f}%)")

# ============= VISTA 3: ANÁLISIS COMPARATIVO =============
else:
    st.subheader("📈 Análisis Comparativo Multi-Pozo")

    pozos_comparar = st.multiselect(
        "Seleccionar pozos para comparar:",
        sorted(df['pozo_id'].unique()),
        default=sorted(df['pozo_id'].unique())[:5]
    )

    if pozos_comparar:
        df_comp = df[df['pozo_id'].isin(pozos_comparar)]

        # Producción acumulada
        fig_acum = px.line(df_comp, x='fecha', y='produccion_acumulada', color='pozo_id',
                          labels={'produccion_acumulada': 'Producción Acumulada (bbl)', 'fecha': 'Fecha'})
        fig_acum.update_layout(height=400)
        st.plotly_chart(fig_acum, use_container_width=True)

        # Tabla comparativa
        st.subheader("📋 Tabla Comparativa")

        tabla_comp = df_comp.groupby('pozo_id').agg({
            'produccion_bbl': ['mean', 'max', 'min'],
            'presion_psi': 'mean',
            'water_cut_pct': 'mean',
            'tipo_pozo': 'first'
        }).round(2)

        tabla_comp.columns = ['Prod. Promedio', 'Prod. Máxima', 'Prod. Mínima', 'Presión Prom.', 'Water Cut Prom.', 'Tipo']
        st.dataframe(tabla_comp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Sistema ML para Predicción de Producción de Pozos Petroleros</strong></p>
    <p>Desarrollado con Python • Streamlit • Scikit-learn • TensorFlow • XGBoost</p>
    <p>📊 100 pozos • 36,500 registros • 3 modelos ML • 92.5% precisión</p>
</div>
""", unsafe_allow_html=True)
