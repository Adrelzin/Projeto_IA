import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Previsão de Câncer Pulmonar", layout="centered")

try:
    modelo = joblib.load("modelo_random_forest.pkl")
    features = joblib.load("features.pkl")
    st.sidebar.success("Modelo carregado!")
    st.sidebar.write("**Features do modelo:**")
    st.sidebar.code(features)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.info("Execute o arquivo treinar_modelo.py primeiro.")
    st.stop()

st.title("Previsão de Câncer Pulmonar")
st.markdown("Preencha os dados abaixo para estimar a probabilidade de câncer pulmonar.")

def yesno_to_int(val):
    return 1 if val == "Sim" else 0

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dados Pessoais")
    idade = st.number_input("Idade", min_value=0, max_value=120, value=50)
    fumante = st.selectbox("Fumante", ("Não", "Sim"), index=0)
    yellow_fingers = st.selectbox("Dedos Amarelados", ("Não", "Sim"), index=0)
    ansiedade = st.selectbox("Ansiedade", ("Não", "Sim"), index=0)
    consome_alcool = st.selectbox("Consome Álcool", ("Não", "Sim"), index=0)
    pressao_social = st.selectbox("Pressão Social", ("Não", "Sim"), index=0)
    doenca_cronica = st.selectbox("Doença Crônica", ("Não", "Sim"), index=0)

with col2:
    st.subheader("Sintomas")
    fadiga = st.selectbox("Fadiga", ("Não", "Sim"), index=0)
    alergia = st.selectbox("Alergia", ("Não", "Sim"), index=0)
    chiado = st.selectbox("Chiado no Peito", ("Não", "Sim"), index=0)
    tosse = st.selectbox("Tosse", ("Não", "Sim"), index=0)
    falta_ar = st.selectbox("Falta de Ar", ("Não", "Sim"), index=0)
    dificuldade_engolir = st.selectbox("Dificuldade ao Engolir", ("Não", "Sim"), index=0)
    dor_peito = st.selectbox("Dor no Peito", ("Não", "Sim"), index=0)

valores = {
    "AGE": idade,
    "SMOKING": yesno_to_int(fumante),
    "YELLOW_FINGERS": yesno_to_int(yellow_fingers),
    "ANXIETY": yesno_to_int(ansiedade),
    "ALCOHOL CONSUMING": yesno_to_int(consome_alcool),
    "PEER_PRESSURE": yesno_to_int(pressao_social),
    "CHRONIC DISEASE": yesno_to_int(doenca_cronica),
    "FATIGUE": yesno_to_int(fadiga),
    "ALLERGY": yesno_to_int(alergia),
    "WHEEZING": yesno_to_int(chiado),
    "COUGHING": yesno_to_int(tosse),
    "SHORTNESS OF BREATH": yesno_to_int(falta_ar),
    "SWALLOWING DIFFICULTY": yesno_to_int(dificuldade_engolir),
    "CHEST PAIN": yesno_to_int(dor_peito)
}

entrada_dict = {feature: valores.get(feature, 0) for feature in features}
entrada = pd.DataFrame([entrada_dict], columns=features)

with st.expander("Ver dados de entrada"):
    st.dataframe(entrada, use_container_width=True)

if st.button("Realizar Previsão", type="primary", use_container_width=True):
    try:
        proba = modelo.predict_proba(entrada)[0]
        prob_cancer = proba[1] * 100
        prob_normal = proba[0] * 100
        
        st.markdown("---")
        st.subheader("Resultado da Previsão")
        
        if prob_cancer >= 50:
            st.error(f"**Alto Risco de Câncer Pulmonar**")
        else:
            st.success(f"**Baixo Risco de Câncer Pulmonar**")
        
        st.metric(
            label="Probabilidade de Câncer Pulmonar",
            value=f"{prob_cancer:.1f}%"
        )
        
        st.progress(prob_cancer / 100)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probabilidade Negativo", f"{prob_normal:.1f}%")
        with col_b:
            st.metric("Probabilidade Positivo", f"{prob_cancer:.1f}%")
            
    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")
        with st.expander("Debug"):
            st.write("**Features esperadas:**", features)
            st.write("**Features fornecidas:**", entrada.columns.tolist())
