import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Previsão de Câncer Pulmonar", layout="centered")

# Carregar modelo e features
try:
    modelo = joblib.load("modelo_random_forest.pkl")
    features = joblib.load("features.pkl")
    st.sidebar.success("✅ Modelo carregado com sucesso!")
    st.sidebar.write("**Features do modelo:**")
    st.sidebar.code(features)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.info("Execute o arquivo modelo.py primeiro para gerar o modelo.")
    st.stop()

st.title("🩺 Previsão de Câncer Pulmonar")
st.markdown("Preencha os dados abaixo para estimar a probabilidade de câncer pulmonar com base no modelo treinado.")

def yesno_to_int(val):
    return 1 if val == "Sim" else 0

# Criar colunas para melhor organização
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dados Pessoais")
    idade = st.number_input("Idade", min_value=0, max_value=120, value=50)
    fumante = st.selectbox("Fumante (Smoking)", ("Não", "Sim"), index=0)
    yellow_fingers = st.selectbox("Dedos Amarelados (Yellow Fingers)", ("Não", "Sim"), index=0)
    ansiedade = st.selectbox("Ansiedade (Anxiety)", ("Não", "Sim"), index=0)
    consome_alcool = st.selectbox("Consome Álcool (Alcohol Consuming)", ("Não", "Sim"), index=0)
    pressao_social = st.selectbox("Pressão Social (Peer Pressure)", ("Não", "Sim"), index=0)
    doenca_cronica = st.selectbox("Doença Crônica (Chronic Disease)", ("Não", "Sim"), index=0)

with col2:
    st.subheader("Sintomas")
    fadiga = st.selectbox("Fadiga (Fatigue)", ("Não", "Sim"), index=0)
    alergia = st.selectbox("Alergia (Allergy)", ("Não", "Sim"), index=0)
    chiado = st.selectbox("Chiado no Peito (Wheezing)", ("Não", "Sim"), index=0)
    tosse = st.selectbox("Tosse (Coughing)", ("Não", "Sim"), index=0)
    falta_ar = st.selectbox("Falta de Ar (Shortness of Breath)", ("Não", "Sim"), index=0)
    dificuldade_engolir = st.selectbox("Dificuldade ao Engolir (Swallowing Difficulty)", ("Não", "Sim"), index=0)
    dor_peito = st.selectbox("Dor no Peito (Chest Pain)", ("Não", "Sim"), index=0)

# Criar dicionário com todos os valores possíveis
valores = {
    "AGE": idade,
    "SMOKING": yesno_to_int(fumante),
    "YELLOW_FINGERS": yesno_to_int(yellow_fingers),
    "ANXIETY": yesno_to_int(ansiedade),
    "ALCOHOL CONSUMING": yesno_to_int(consome_alcool),
    "PEER_PRESSURE": yesno_to_int(pressao_social),
    "CHRONIC DISEASE": yesno_to_int(doenca_cronica),
    "FATIGUE ": yesno_to_int(fadiga),  # Pode ter espaço extra
    "FATIGUE": yesno_to_int(fadiga),   # Sem espaço
    "ALLERGY ": yesno_to_int(alergia), # Pode ter espaço extra
    "ALLERGY": yesno_to_int(alergia),  # Sem espaço
    "WHEEZING": yesno_to_int(chiado),
    "COUGHING": yesno_to_int(tosse),
    "SHORTNESS OF BREATH": yesno_to_int(falta_ar),
    "SWALLOWING DIFFICULTY": yesno_to_int(dificuldade_engolir),
    "CHEST PAIN": yesno_to_int(dor_peito)
}

# CRÍTICO: Montar DataFrame na MESMA ORDEM que o modelo foi treinado
entrada_dict = {}
for feature in features:
    # Tentar encontrar o valor, considerando possíveis espaços
    valor = valores.get(feature, valores.get(feature.strip(), 0))
    entrada_dict[feature] = valor

# Criar DataFrame com colunas na ordem correta
entrada = pd.DataFrame([entrada_dict], columns=features)

with st.expander("📊 Ver dados de entrada"):
    st.write("**Features esperadas pelo modelo:**")
    st.code(features)
    st.write("**Dados enviados:**")
    st.dataframe(entrada, use_container_width=True)

# Previsão
if st.button("🔍 Realizar Previsão", type="primary", use_container_width=True):
    try:
        # Verificar se o modelo tem predict_proba (classificador)
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(entrada)[0]
            prob_positivo = proba[1] * 100
            prob_negativo = proba[0] * 100
            
            # Exibir resultado
            st.markdown("---")
            st.subheader("📋 Resultado da Previsão")
            
            # Criar visualização com cores
            if prob_positivo >= 50:
                st.error(f"⚠️ **Alto Risco de Câncer Pulmonar**")
                st.metric(
                    label="Probabilidade de Câncer Pulmonar",
                    value=f"{prob_positivo:.1f}%",
                    delta=f"{prob_positivo - 50:.1f}% acima do limite"
                )
            else:
                st.success(f"✅ **Baixo Risco de Câncer Pulmonar**")
                st.metric(
                    label="Probabilidade de Câncer Pulmonar",
                    value=f"{prob_positivo:.1f}%",
                    delta=f"{50 - prob_positivo:.1f}% abaixo do limite"
                )
            
            # Barra de progresso visual
            st.progress(prob_positivo / 100)
            
            # Mostrar ambas as probabilidades
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"🔵 Negativo: {prob_negativo:.1f}%")
            with col_b:
                st.info(f"🔴 Positivo: {prob_positivo:.1f}%")
            
            # Aviso importante
            st.warning("⚠️ **Aviso Importante**: Este resultado é apenas uma estimativa baseada em dados estatísticos. "
                      "Não substitui avaliação médica profissional. Consulte um médico para diagnóstico adequado.")
        else:
            # Fallback para modelos sem predict_proba
            pred = modelo.predict(entrada)[0]
            if pred >= 0.5:
                st.error(f"⚠️ **Resultado: Alto Risco** (valor: {pred:.2f})")
            else:
                st.success(f"✅ **Resultado: Baixo Risco** (valor: {pred:.2f})")
            
    except Exception as e:
        st.error(f"❌ Erro ao realizar a previsão: {e}")
        st.info("Verifique se o modelo foi treinado corretamente e se todas as features estão presentes.")
        
        # Debug info
        with st.expander("🔍 Informações de Debug"):
            st.write("**Features esperadas:**", features)
            st.write("**Features fornecidas:**", entrada.columns.tolist())
            st.write("**Entrada completa:**")
            st.dataframe(entrada)