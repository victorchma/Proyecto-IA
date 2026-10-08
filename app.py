import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def cargar():
    art = joblib.load("modelo.joblib")
    return art["modelo"], art["umbral"]

modelo, UMBRAL = cargar()

st.markdown("""
<style>
.stApp {
    background-color: #E9F2FF;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;
color:#CF7891;
font-size:42px;
font-family:Georgia;
font-weight:700;'>
🧠 Modelo Predictivo de Riesgo de Ictus
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
text-align: justify;
font-size:18px;
color:#444444;
line-height:1.7;
max-width:900px;
margin:auto;'>

Esta aplicación estima el <b>riesgo de sufrir un ictus</b> a partir de información
demográfica, clínica y de hábitos de vida del paciente mediante un modelo de
<b>Machine Learning (SVM)</b>.

Complete el siguiente formulario y pulse
<b>"Calcular riesgo de ictus"</b>
para obtener la predicción.

</div>
""", unsafe_allow_html=True)

with st.form("paciente"):

    st.subheader("👤 Datos personales")

    age = st.slider(
        "Edad",
        min_value=0,
        max_value=100,
        value=50,
        help="Edad del paciente en años."
    )

    gender = st.selectbox(
        "Sexo",
        ["Hombre", "Mujer"],
        help="Seleccione el sexo biológico del paciente."
    )

    st.subheader("📋 Antecedentes médicos")

    hypertension = st.selectbox(
        "Hipertensión arterial",
        ["No", "Sí"],
        help="Indique si el paciente ha sido diagnosticado de hipertensión arterial."
    )

    heart_disease = st.selectbox(
        "Enfermedad cardíaca",
        ["No", "Sí"],
        help="Indique si el paciente presenta antecedentes de enfermedad cardíaca."
    )

    ever_married = st.selectbox(
        "¿Ha estado casado/a alguna vez?",
        ["No", "Sí"],
        help="Estado civil histórico del paciente."
    )

    st.subheader("🏠 Información sociodemográfica")

    work_type = st.selectbox(
        "Situación laboral",
        [
            "Empleado en empresa privada",
            "Trabajador autónomo",
            "Empleado público",
            "Menor de edad",
            "Nunca ha trabajado"
        ],
        help="Seleccione la situación laboral del paciente."
    )

    Residence_type = st.selectbox(
        "Tipo de residencia",
        ["Urbana", "Rural"],
        help="Lugar habitual de residencia del paciente."
    )

    smoking_status = st.selectbox(
        "Hábito tabáquico",
        [
            "Nunca ha fumado",
            "Exfumador",
            "Fumador",
            "Desconocido"
        ],
        help="Seleccione el historial de tabaquismo del paciente."
    )

    st.subheader("🩺 Datos clínicos")

    avg_glucose_level = st.number_input(
        "Nivel medio de glucosa en sangre (mg/dL)",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        step=1.0,
        help="Valor medio de glucosa en sangre del paciente."
    )

    st.markdown("##### Peso y altura")

    st.write(
        "Introduzca el peso y la altura del paciente. "
        "La aplicación calculará automáticamente el índice de masa corporal (IMC), "
        "una de las variables utilizadas por el modelo predictivo."
    )

    conoce_bmi = st.checkbox(
        "Conozco el peso y la altura del paciente",
        value=True,
        help="Desmarque esta opción si desconoce alguno de estos datos."
    )

    if conoce_bmi:

        peso = st.number_input(
            "Peso (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
            help="Peso del paciente en kilogramos."
        )

        altura = st.number_input(
            "Altura (cm)",
            min_value=100.0,
            max_value=230.0,
            value=170.0,
            step=1.0,
            help="Altura del paciente en centímetros."
        )

        # Cálculo automático del IMC
        bmi = peso / ((altura / 100) ** 2)

        st.success(f"IMC calculado automáticamente: **{bmi:.1f} kg/m²**")

    else:

        bmi = np.nan

    enviar = st.form_submit_button("Calcular riesgo de ictus")

if enviar:

    # Conversión de los valores introducidos por el usuario

    gender = {
        "Hombre": "Male",
        "Mujer": "Female"
    }[gender]

    hypertension = {
        "No": 0,
        "Sí": 1
    }[hypertension]

    heart_disease = {
        "No": 0,
        "Sí": 1
    }[heart_disease]

    ever_married = {
        "No": "No",
        "Sí": "Yes"
    }[ever_married]

    work_type = {
        "Empleado en empresa privada": "Private",
        "Trabajador autónomo": "Self-employed",
        "Empleado público": "Govt_job",
        "Menor de edad": "children",
        "Nunca ha trabajado": "Never_worked"
    }[work_type]

    Residence_type = {
        "Urbana": "Urban",
        "Rural": "Rural"
    }[Residence_type]

    smoking_status = {
        "Nunca ha fumado": "never smoked",
        "Exfumador": "formerly smoked",
        "Fumador": "smokes",
        "Desconocido": "Unknown"
    }[smoking_status]

    # Creación de las variables derivadas

    bmi_missing = int(np.isnan(bmi))

    comorbidity_count = hypertension + heart_disease

    age_x_hypertension = age * hypertension

    age_x_heart = age * heart_disease

    age_x_glucose = age * avg_glucose_level / 100

    is_elderly = int(age >= 60)

    is_child = int(age < 18)

    glucose_high = int(avg_glucose_level >= 126)

    obese = int(not np.isnan(bmi) and bmi >= 30)

    # Variables categóricas creadas durante el entrenamiento

    if age < 18:
        age_group = "menor"
    elif age < 40:
        age_group = "adulto_joven"
    elif age < 60:
        age_group = "adulto"
    else:
        age_group = "senior"

    if avg_glucose_level < 100:
        glucose_category = "normal"
    elif avg_glucose_level < 126:
        glucose_category = "prediabetes"
    else:
        glucose_category = "diabetes"

    if np.isnan(bmi):
        bmi_category = "desconocido"
    elif bmi < 18.5:
        bmi_category = "bajo_peso"
    elif bmi < 25:
        bmi_category = "normal"
    elif bmi < 30:
        bmi_category = "sobrepeso"
    else:
        bmi_category = "obesidad"

    # Construcción del ejemplo para el modelo

    ejemplo = pd.DataFrame([{

        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,

        "bmi_missing": bmi_missing,
        "glucose_category": glucose_category,
        "bmi_category": bmi_category,
        "age_group": age_group,
        "comorbidity_count": comorbidity_count,
        "age_x_hypertension": age_x_hypertension,
        "age_x_heart": age_x_heart,
        "age_x_glucose": age_x_glucose,
        "is_elderly": is_elderly,
        "is_child": is_child,
        "glucose_high": glucose_high,
        "obese": obese

    }])

    # --- PREDICCIÓN DEL MODELO ---

    proba = float(modelo.predict_proba(ejemplo)[:, 1][0])

    alerta = proba >= UMBRAL

    # Mostrar resultados

    st.divider()

    st.subheader("📊 Resultado de la predicción")

    st.metric(
        label="Probabilidad estimada de sufrir un ictus",
        value=f"{proba*100:.1f}%"
    )

    if alerta:

        st.error("⚠️ Riesgo elevado de sufrir un ictus")

        st.write(
            f"El modelo estima una **probabilidad del {proba*100:.1f}%** "
            "de sufrir un ictus. Como este valor supera el umbral de decisión "
            f"establecido ({UMBRAL*100:.1f}%), el paciente se clasifica como "
            "**caso de riesgo elevado**."
        )

    else:

        st.success("✅ Riesgo bajo de sufrir un ictus")

        st.write(
            f"El modelo estima una **probabilidad del {proba*100:.1f}%** "
            "de sufrir un ictus. Como este valor es inferior al umbral de decisión "
            f"({UMBRAL*100:.1f}%), el paciente se clasifica como "
            "**caso de riesgo bajo**."
        )

    st.caption(
    f"Umbral de decisión utilizado por el modelo: {UMBRAL*100:.1f}% "
    "(seleccionado durante la fase de validación para priorizar la detección de casos de ictus)."
    )
    st.info(
    "ℹ️ Este resultado es una estimación obtenida mediante un modelo de "
    "**Machine Learning (SVM)** entrenado con datos históricos.\n\n"
    "**⚠️ NO SUSTITUYE EL DIAGNÓSTICO NI LA VALORACIÓN DE UN PROFESIONAL SANITARIO.**"
    )

    with st.expander("Ver información utilizada para la predicción"):
        st.dataframe(ejemplo)