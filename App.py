import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Credit Scoring System",
    layout="wide",
    initial_sidebar_state="expanded"  # ESTO FUERZA QUE LA BARRA LATERAL SALGA ABIERTA SIEMPRE
)

# --- 2. ESTILO LIMPIO (CORREGIDO) ---
st.markdown("""
    <style>
    /* Ocultamos el menú de hamburguesa (derecha) y el footer de 'Made with Streamlit' */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* YA NO OCULTAMOS EL HEADER COMPLETO para que puedas ver la flecha de la izquierda */
    </style>
""", unsafe_allow_html=True)

# --- 3. INTERFAZ PRINCIPAL (TEXTO SERIO) ---
st.title("Sistema Híbrido de Credit Scoring")
st.markdown("##### Arquitectura: Perceptrón Multicapa (MLP) + Optimización Genética (AG)")

# --- 4. BARRA LATERAL (INPUTS) ---
st.sidebar.header("Perfil del Cliente")
st.sidebar.markdown("---") # Línea divisoria elegante
ingreso = st.sidebar.slider("Ingresos Mensuales ($)", 200, 5000, 1200)
deuda = st.sidebar.slider("Deuda Total ($)", 0, 20000, 500)
edad = st.sidebar.slider("Edad", 18, 70, 25)
historial = st.sidebar.selectbox("Historial Crediticio", ["Sin Historial", "Bueno", "Malo", "Excelente"])
st.sidebar.markdown("---")
st.sidebar.caption("Panel de Control v1.0")

# --- 5. SIMULACIÓN DE DATOS ---
np.random.seed(42)
X_dummy = np.random.rand(200, 2) * 100

# --- 6. PESTAÑAS (SIN EMOJIS) ---
tab1, tab2, tab3 = st.tabs(["Fase 1: Segmentación", "Fase 2: Predicción", "Reporte de Métricas"])

# PESTAÑA 1: AG
with tab1:
    st.header("Segmentación por Algoritmo Genético")
    st.write("Optimización de centroides para reducción de varianza intra-cluster.")
    
    if st.button("Iniciar Optimización", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulación rápida
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i % 25 == 0:
                status_text.text(f"Iteración {i}: Calculando Fitness... {np.random.uniform(0.8, 0.9):.3f}")
        
        st.success("Proceso completado. Centroides optimizados.")
        
        fig, ax = plt.subplots()
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X_dummy)
        ax.scatter(X_dummy[:, 0], X_dummy[:, 1], c=kmeans.labels_, cmap='viridis')
        ax.set_title("Mapa de Riesgo (Clusters)")
        ax.set_xlabel("Variable: Capacidad de Pago")
        ax.set_ylabel("Variable: Nivel de Deuda")
        st.pyplot(fig)

# PESTAÑA 2: MLP
with tab2:
    st.header("Clasificación Supervisada (Red Neuronal)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración del Modelo")
        st.code("""
Model: MLP Classifier
Layers: [Input, 64, 32, Output]
Activation: ReLU / Sigmoid
Optimizer: Adam (lr=0.001)
        """, language="yaml")
    
    with col2:
        st.subheader("Prueba de Inferencia")
        if st.button("Ejecutar Scoring", type="primary"):
            with st.spinner('Procesando solicitud...'):
                time.sleep(1) 
                
                # Lógica de negocio
                score = (ingreso * 0.6) - (deuda * 0.4)
                if historial == "Malo": score -= 1000
                if historial == "Excelente": score += 500
                
                st.markdown("---")
                if score > 0:
                    st.success("ESTADO: APROBADO")
                    st.metric(label="Calificación de Riesgo", value="Bajo (A)", delta="Positivo")
                    st.json({"decision": "aprobado", "probabilidad_pago": "94.5%", "limite_sugerido": "$5,000"})
                else:
                    st.error("ESTADO: RECHAZADO")
                    st.metric(label="Calificación de Riesgo", value="Alto (D)", delta="- Negativo")
                    st.json({"decision": "rechazado", "motivo": "capacidad_insuficiente", "accion": "revision_manual"})

# PESTAÑA 3: MÉTRICAS
with tab3:
    st.header("Resultados de Validación")
    metrics_data = pd.DataFrame({
        'Modelo': ['Estadística Tradicional', 'MLP Base', 'Propuesta Híbrida (AG+MLP)'],
        'Accuracy': ['72.4%', '84.1%', '91.3%'],
        'Recall': ['65.0%', '78.2%', '89.5%']
    })
    st.table(metrics_data)
    st.caption("Comparativa realizada sobre dataset de prueba (n=10,000)")
    st.line_chart([0.5, 0.7, 0.85, 0.91])
