import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image

# --- Nueva Función para Crear el Gráfico Estático (para Excel) ---
def create_matplotlib_figure(df, results, threshold):
    """Crea una figura estática con Matplotlib para el reporte."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time_s'], df['current_kA'], label='Señal Original i(t)', alpha=0.7, color='royalblue')
    ax.plot(df['time_s'], df['i_rms'], label='RMS Móvil i_rms(t)', color='red')
    
    ax.axhline(y=threshold, color='green', linestyle='--', label='Umbral')
    
    # Extraer valores de los resultados para las líneas verticales
    t_inicio_evento = results.loc[results['Parámetro'] == 'Tiempo de Inicio del Evento (s)', 'Valor'].iloc[0]
    t_pico_energia = results.loc[results['Parámetro'] == 'Tiempo del Pico de Energía (s)', 'Valor'].iloc[0]
    t_inicio_fusion = results.loc[results['Parámetro'] == 'Tiempo de Inicio de Fusión (s)', 'Valor'].iloc[0]
    t_fin_evento = results.loc[results['Parámetro'] == 'Tiempo de Fin del Evento (s)', 'Valor'].iloc[0]

    # Se convierte a float para asegurar que Matplotlib pueda graficarlo aunque sea N/A
    if pd.notna(t_inicio_evento): ax.axvline(x=float(t_inicio_evento), color='purple', linestyle='--', label='Inicio Evento')
    if pd.notna(t_pico_energia): ax.axvline(x=float(t_pico_energia), color='orange', linestyle='--', label='Pico de Energía')
    if pd.notna(t_inicio_fusion): ax.axvline(x=float(t_inicio_fusion), color='brown', linestyle='--', label='Inicio Fusión')
    if pd.notna(t_fin_evento): ax.axvline(x=float(t_fin_evento), color='cyan', linestyle='--', label='Fin Evento')
        
    ax.set_title('Análisis de Señal de Fusión de Fusible')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Corriente (kA)')
    ax.legend(title='Componentes del Gráfico')
    ax.grid(True)
    return fig

# --- Función para Crear el Reporte de Excel ---
def create_excel_report(df_results, matplotlib_fig):
    """Genera un archivo de Excel con la tabla y el gráfico estático."""
    output_buffer = io.BytesIO()
    
    # Guardar el gráfico de Matplotlib en un buffer de imagen
    img_buffer = io.BytesIO()
    matplotlib_fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img = OpenpyxlImage(img_buffer)

    # Crear el libro de trabajo de Excel
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Resultados del Análisis', index=False, startrow=1)
        worksheet = writer.sheets['Resultados del Análisis']
        
        # Formatear la hoja
        worksheet.cell(row=1, column=1, value="Resultados del Análisis de Fusión")
        worksheet.column_dimensions['A'].width = 40
        worksheet.column_dimensions['B'].width = 25
        
        # Insertar la imagen
        worksheet.add_image(img, 'D2')

    return output_buffer.getvalue()


# --- Función de Análisis Principal ---
def analyze_signal(df, N, k):
    # 1. Preparación y Detección de Evento
    df.columns = ['time_ms', 'current_kA']
    df['time_s'] = df['time_ms'] / 1000.0
    noise_period_end_time_s = df['time_s'].quantile(0.3)
    noise_data = df[df['time_s'] <= noise_period_end_time_s]
    i_rms_base = np.sqrt(np.mean(noise_data['current_kA']**2))
    df['i_rms'] = np.sqrt(df['current_kA'].rolling(window=N, center=True).apply(lambda x: np.mean(x**2), raw=True).fillna(0))
    threshold = k * i_rms_base
    df['is_event'] = (df['i_rms'] > threshold).astype(int)
    event_start_indices = np.where(df['is_event'].diff() == 1)[0]
    t_inicio_evento = df['time_s'].iloc[event_start_indices[0]] if len(event_start_indices) > 0 else None
    event_end_indices = np.where(df['is_event'].diff() == -1)[0]
    t_fin_evento = None
    if t_inicio_evento is not None and len(event_end_indices) > 0:
        end_indices_after_start = event_end_indices[event_end_indices > event_start_indices[0]]
        if len(end_indices_after_start) > 0:
            t_fin_evento = df['time_s'].iloc[end_indices_after_start[0]]

    # 2. Identificación de Fases
    t_pico_energia, t_inicio_fusion = None, None
    I_rms_trans_inicial, I_rms_estable, I_rms_trans_final = None, None, None
    if t_inicio_evento and t_fin_evento:
        event_period_df = df[(df['time_s'] >= t_inicio_evento) & (df['time_s'] <= t_fin_evento)]
        if not event_period_df.empty:
            peak_rms_index = event_period_df['i_rms'].idxmax()
            t_pico_energia = df['time_s'].loc[peak_rms_index]
            peak_rms_value = df['i_rms'].loc[peak_rms_index]
            fusion_trigger_level = peak_rms_value * 0.85
            after_peak_df = event_period_df.loc[peak_rms_index:]
            fusion_candidates = after_peak_df[after_peak_df['i_rms'] < fusion_trigger_level]
            if not fusion_candidates.empty:
                t_inicio_fusion = df['time_s'].loc[fusion_candidates.index[0]]
            if t_pico_energia and t_inicio_fusion:
                trans_inicial_data = df[(df['time_s'] >= t_inicio_evento) & (df['time_s'] < t_pico_energia)]
                if not trans_inicial_data.empty: I_rms_trans_inicial = np.sqrt(np.mean(trans_inicial_data['current_kA']**2))
                estable_data = df[(df['time_s'] >= t_pico_energia) & (df['time_s'] < t_inicio_fusion)]
                if not estable_data.empty: I_rms_estable = np.sqrt(np.mean(estable_data['current_kA']**2))
                trans_final_data = df[(df['time_s'] >= t_inicio_fusion) & (df['time_s'] <= t_fin_evento)]
                if not trans_final_data.empty: I_rms_trans_final = np.sqrt(np.mean(trans_final_data['current_kA']**2))

    # 3. Crear DataFrame de Resultados
    results_df = pd.DataFrame({
        'Parámetro': ['Ruido RMS de Base (A)', 'Umbral Adaptativo (A)', 'Tiempo de Inicio del Evento (s)', 'Tiempo del Pico de Energía (s)', 'Tiempo de Inicio de Fusión (s)', 'Tiempo de Fin del Evento (s)', 'Corriente RMS Transitorio Inicial (A)', 'Corriente RMS Estado Estable (A)', 'Corriente RMS Transitorio Final (A)'],
        'Valor': [i_rms_base, threshold, t_inicio_evento, t_pico_energia, t_inicio_fusion, t_fin_evento, I_rms_trans_inicial, I_rms_estable, I_rms_trans_final]
    })
    results_df['Valor'] = results_df['Valor'].apply(lambda x: f'{x:.7f}' if isinstance(x, (int, float)) else 'N/A')
    
    # 4. Crear Gráfico Interactivo de Plotly
    plotly_fig = go.Figure()
    plotly_fig.add_trace(go.Scatter(x=df['time_s'], y=df['current_kA'], mode='lines', name='Señal Original i(t)', line=dict(color='royalblue'), hovertemplate='...'))
    plotly_fig.add_trace(go.Scatter(x=df['time_s'], y=df['i_rms'], mode='lines', name='RMS Móvil i_rms(t)', line=dict(color='red'), hoverinfo='skip'))
    y_range = [df['current_kA'].min(), df['current_kA'].max()]
    plotly_fig.add_trace(go.Scatter(x=[df['time_s'].iloc[0], df['time_s'].iloc[-1]], y=[threshold, threshold], mode='lines', name='Umbral', line=dict(color='green', dash='dash')))
    if t_inicio_evento: plotly_fig.add_trace(go.Scatter(x=[t_inicio_evento, t_inicio_evento], y=y_range, mode='lines', name='Inicio Evento', line=dict(color='purple', dash='dash')))
    if t_pico_energia: plotly_fig.add_trace(go.Scatter(x=[t_pico_energia, t_pico_energia], y=y_range, mode='lines', name='Pico de Energía', line=dict(color='orange', dash='dash')))
    if t_inicio_fusion: plotly_fig.add_trace(go.Scatter(x=[t_inicio_fusion, t_inicio_fusion], y=y_range, mode='lines', name='Inicio Fusión', line=dict(color='brown', dash='dash')))
    if t_fin_evento: plotly_fig.add_trace(go.Scatter(x=[t_fin_evento, t_fin_evento], y=y_range, mode='lines', name='Fin Evento', line=dict(color='cyan', dash='dash')))
    plotly_fig.update_layout(title='Análisis Interactivo de Señal de Fusión de Fusible', xaxis_title='Tiempo (s)', yaxis_title='Corriente (kA)', legend_title='Componentes del Gráfico', hovermode='x unified')

    # CORRECCIÓN: Devolver el DataFrame analizado junto con los otros resultados
    return plotly_fig, results_df, threshold, df

# --- Interfaz de la Aplicación Streamlit ---
st.set_page_config(layout="wide")
st.title("Herramienta de Análisis de Fusión de Fusibles")
st.write("Sube tu archivo CSV para identificar automáticamente las fases de transitorio inicial, estado estable y fusión.")
with st.sidebar:
    st.header("Parámetros de Análisis")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    N = st.slider("Tamaño de Ventana (N)", 10, 200, 50, help="...")
    k = st.slider("Factor de Umbral (k)", 1.0, 15.0, 5.0, 0.5, help="...")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # CORRECCIÓN: Recibir el DataFrame analizado de la función
        plotly_fig, results_df, threshold, analyzed_df = analyze_signal(df.copy(), N, k)
        
        st.header("Resultados del Análisis")
        
        # CORRECCIÓN: Usar el DataFrame analizado (analyzed_df) para crear la figura estática
        matplotlib_fig = create_matplotlib_figure(analyzed_df, results_df, threshold)
        # Generar el archivo de Excel en memoria
        excel_data = create_excel_report(results_df, matplotlib_fig)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(results_df)
            st.download_button(
                label="Descargar Reporte en Excel",
                data=excel_data,
                file_name="reporte_analisis_fusion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            st.plotly_chart(plotly_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Ocurrió un error durante el análisis: {e}")
else:
    st.info("Esperando a que se suba un archivo CSV.")

