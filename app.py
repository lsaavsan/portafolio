# Importamos las bibliotecas necesarias para la aplicación
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import base64
import seaborn as sns
import plotly.graph_objects as go

# Configuración de la página de la aplicación en Streamlit
st.set_page_config(
    page_title="Mi Aplicación",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definimos el título y la descripción de la aplicación
st.title('Clasificando especies de Pingüinos')

# Ruta de la imagen del logo
logo = 'especies.jpg'
st.image(logo, width=1000)

# Texto descriptivo sobre los pingüinos
st.markdown("""
Los pingüinos son aves fascinantes que habitan principalmente en el hemisferio sur, con especies que viven en diversos hábitats, desde las frías costas de la Antártida hasta islas más templadas en el sur del continente americano. A diferencia de la mayoría de las aves, los pingüinos han adaptado su anatomía para un estilo de vida acuático. Sus alas han evolucionado en aletas que les permiten nadar con gran habilidad, mientras que sus cuerpos robustos y su densa capa de plumas los protegen del frío.
""")
st.markdown("""____""")

# Cargar y preparar el conjunto de datos de pingüinos para visualización
df = sns.load_dataset("penguins")

# Crear un gráfico de caja para visualizar la masa corporal por especie y sexo de los pingüinos
fig = px.box(
    df,
    x="species",
    y="body_mass_g",
    color="sex",
    color_discrete_sequence=['red','blue'],  # Aplicar la secuencia de colores personalizada
    title="Distribución de Masa Corporal por Especie y Sexo de Pingüinos",
    labels={
        "species": "Especie",
        "body_mass_g": "Masa Corporal (g)",
        "sex": "Sexo"
    },
    notched=False 
)
fig.update_layout(
    width=1000,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50),
    title_x=0.2
)
st.plotly_chart(fig)
st.markdown("""---""")

# Crear un gráfico de violín para analizar la longitud del pico según la isla y la especie
fig = px.violin(
    df,
    x="island",
    y="bill_length_mm",
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    title="Distribución de Longitud del Pico por Isla y Especie de Pingüinos",
    labels={
        "island": "Islas",
        "bill_length_mm": "Longitud Pico (mm)",
        "species": "Especies"
    },
    box=True
)
fig.update_layout(
    width=1000,
    height=600,
    margin=dict(l=50, r=50, b=50, t=50),
    title_x=0.2
)
st.plotly_chart(fig)
st.markdown("""---""")

# Crear una matriz de dispersión para analizar la relación entre características físicas de los pingüinos
fig = px.scatter_matrix(
    df,
    dimensions=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    title="Matriz de Dispersión de las características físicas de los Pingüinos",
    symbol="sex",
    labels={
        "bill_length_mm": "Longitud Pico (mm)",
        "bill_depth_mm": "Profundidad Pico (mm)",
        "flipper_length_mm": "Longitud Aleta (mm)",
        "body_mass_g": "Masa Corporal (g)"
    }
)
fig.update_layout(
    width=1000,
    height=1200,
    margin=dict(l=50, r=50, b=50, t=50),
    title_x=0.2
)
st.plotly_chart(fig)
st.markdown("""---""")

# Visualizar la longitud de la aleta ordenada de menor a mayor para cada especie
species_counts = df[['flipper_length_mm','species']].sort_values(by='flipper_length_mm').reset_index()
species_counts['index'] = range(len(species_counts))
fig = px.line(
    species_counts,
    x="index",
    y="flipper_length_mm",
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    title="Crecimiento de Longitud de Aleta (Ordenado de Menor a Mayor) por Especie de Pingüino",
    labels={"index": "Índice (Ordenado)", "flipper_length_mm": "Longitud de Aleta (mm)"}
)
fig.update_layout(width=1000, height=600, margin=dict(l=50, r=50, b=50, t=50), title_x=0.2)
st.plotly_chart(fig)
st.markdown("""---""")

# Crear un gráfico de pastel para mostrar la distribución de pingüinos por especie
species_counts = df['species'].value_counts().reset_index()
species_counts.columns = ['species', 'count']
fig = px.pie(species_counts, names='species', values='count', title="Distribución de Pingüinos por Especie",color_discrete_sequence=['red','blue','green'])
fig.update_layout(width=500, height=500, margin=dict(l=50, r=50, b=50, t=50), title_x=0.0)
st.plotly_chart(fig)
st.markdown("""---""")

# Crear un mapa de calor de correlación entre variables numéricas
correlation_matrix = df.corr(numeric_only=True)
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale="Viridis",
    colorbar=dict(title="Correlación")
))
fig.update_layout(title="Mapa de Calor de Correlación entre Variables Numéricas", width=800, height=800, margin=dict(l=50, r=50, b=50, t=50), title_x=0.2)
st.plotly_chart(fig)
st.markdown("""---""")

# Crear un histograma de la longitud del pico por especie
fig = px.histogram(
    df,
    x="bill_length_mm",
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    title="Distribución de Longitud del Pico (bill_length_mm) por Especie",
    labels={"bill_length_mm": "Longitud del Pico (mm)", "species": "Especie"},
    nbins=80,
    barmode="overlay"
)
fig.update_layout(width=1000, height=600, margin=dict(l=50, r=50, b=50, t=50), title_x=0.2)
st.plotly_chart(fig)
st.markdown("""---""")

# Calcular la media de la longitud del pico por especie y sexo y mostrarla en un gráfico de barras
species_sex_medias = df.groupby(["species", "sex"])["bill_length_mm"].mean().reset_index()
fig = px.bar(
    species_sex_medias,
    x="sex",
    y="bill_length_mm",
    title="Longitud del Pico por Especie y Sexo",
    labels={"species": "Especie", "bill_length_mm": "Longitud del Pico (mm)", "sex": "Sexo"},
    color='species',
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    barmode="group",
)
fig.update_layout(width=1000, height=600, margin=dict(l=50, r=50, b=50, t=50), title_x=0.2)
st.plotly_chart(fig)
st.markdown("""---""")

# Gráfico de densidad de contorno
fig = px.density_contour(
    df,
    x="body_mass_g",
    y="flipper_length_mm",
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    title="Densidad de Masa Corporal y Longitud de Aleta por Especie",
    labels={"body_mass_g": "Masa Corporal (g)", "flipper_length_mm": "Longitud de Aleta (mm)"}
)
fig.update_layout(width=800, height=500, title_x=0.2)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.markdown("""---""")
# Gráfico de dispersión con burbujas
df_clean = df.dropna()
fig = px.scatter(
    df_clean,
    x="bill_length_mm",
    y="body_mass_g",
    color="species",
    color_discrete_sequence=['red','blue','green'],  # Aplicar la secuencia de colores personalizada
    size="flipper_length_mm",
    symbol="island",
    title="Relación entre Longitud del Pico y Masa Corporal, por Especie y Isla donde se observó",
    labels={"bill_length_mm": "Longitud del Pico (mm)", "body_mass_g": "Masa Corporal (g)"},
    size_max=5  # Ajuste el tamaño máximo de las burbujas
)
fig.update_layout(width=800, height=500, title_x=0.2)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.markdown("""---""")




# Configuración del panel lateral en Streamlit para la entrada de datos del usuario
st.sidebar.header('Datos')

# Cargar modelo, codificador y escalador preentrenados
load_clf = pickle.load(open('rf_pinguinos.pkl', 'rb'))
encoder = pickle.load(open('encoder_pinguinos.pkl', 'rb'))
scaler = pickle.load(open('scaler_pinguinos.pkl', 'rb'))
X_train_prepared = pickle.load(open('X_train_prepared_pinguinos.pkl', 'rb'))

# Definir función para capturar las características del usuario desde la interfaz de Streamlit
def user_input_features():
    island = st.sidebar.selectbox('Isla donde se identificó al pingüino', ('Torgersen', 'Biscoe', 'Dream'))
    sex = st.sidebar.selectbox('Sexo del pingüino', ('Male', 'Female'))
    bill_length_mm = st.sidebar.slider('Longitud del pico en mm', 32, 59, 44)
    bill_depth_mm = st.sidebar.slider('Profundidad del pico en mm', 13, 21, 17)
    flipper_length_mm = st.sidebar.slider('Longitud de la aleta en mm', 172, 231, 197)
    body_mass_g = st.sidebar.slider('Masa corporal en gramos', 2700, 6300, 4050)

    # Crear DataFrame con las características ingresadas por el usuario
    data = {
        'island': [island],
        'sex': [sex],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    }
    return pd.DataFrame(data, index=[0])

# Permitir al usuario cargar un archivo de datos o ingresar manualmente las características
uploaded_file = st.sidebar.file_uploader("Subir un archivo de Excel con las características a clasificar", type="xlsx")
input_df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file else user_input_features()

# Codificar las características categóricas con el codificador cargado
categorical_features = input_df[['island', 'sex']]
encoded_features = encoder.transform(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['island', 'sex']))

# Escalar las características numéricas con el escalador cargado
numerical_features = input_df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
scaled_numerical = scaler.transform(numerical_features)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)

# Combinar datos procesados en un solo DataFrame y alinear con `X_train_prepared`
input_df_processed = pd.concat([scaled_numerical_df, encoded_df], axis=1)
input_df_processed = input_df_processed.reindex(columns=X_train_prepared.columns, fill_value=0)

# Mostrar los datos ingresados por el usuario
st.subheader('Características para la Clasificación de Pingüinos')
st.write(input_df)

# Realizar la predicción con el modelo cargado
prediction = load_clf.predict(input_df_processed)
prediction_proba = load_clf.predict_proba(input_df_processed)

# Mostrar resultados de la predicción y la probabilidad
col1, col2 = st.columns(2)
with col1:
    st.subheader('Predicción')
    if prediction[0] == 0:
        st.write('Especie Adelie')
        logo0 = 'Adelie.png'
        st.image(logo0, width=300)
        st.markdown("""
        Conocidos por su característica franja blanca alrededor de los ojos, son nativos de la costa antártica. Su nombre proviene de la esposa del explorador francés Jules Dumont d'Urville, quien los nombró en honor a su esposa, Adèle.
        """)
    elif prediction[0] == 2:
        st.write('Especie Gentoo')
        logo1 = 'Gentoo.png'
        st.image(logo1, width=300)
        st.markdown("""
        Reconocible por su pico naranja brillante y una mancha blanca en la parte superior de su cabeza. Son los más rápidos nadadores entre los pingüinos y habitan en diversas islas subantárticas.
        """)
    else:
        st.write('Especie Chinstrap')
        logo2 = 'Chinstrap.png'
        st.image(logo2, width=300)
        st.markdown("""
        También llamado barbijo, por la "barbilla" negra que se asemeja a una correa debajo de su pico. Son excelentes nadadores y habitan en islas y costas del Antártico.
        """)

with col2:
    st.subheader('Probabilidad de predicción')
    st.write(prediction_proba)

st.markdown("""---""")
