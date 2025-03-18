import pandas as pd
import numpy as np
import json
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Simulação de um banco de dados de imóveis
imoveis = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'localizacao': ['Lisboa', 'Porto', 'Lisboa', 'Algarve', 'Porto'],
    'preco': [300000, 250000, 400000, 200000, 270000],
    'quartos': [3, 2, 4, 2, 3],
    'garagem': [1, 0, 1, 1, 0],
    'piscina': [0, 0, 1, 1, 0]
})

# Simulação de um perfil de usuário
usuario = {
    'localizacao': 'Lisboa',
    'preco_max': 350000,
    'quartos': 3,
    'garagem': 1,
    'piscina': 0
}

# Carregar feedback do usuário de um arquivo JSON
feedback_file = "feedback.json"
try:
    with open(feedback_file, "r") as f:
        feedback_historico = json.load(f)
except FileNotFoundError:
    feedback_historico = {}

# Filtrar imóveis por localização e preço máximo
imoveis_filtrados = imoveis[(imoveis['localizacao'] == usuario['localizacao']) & (imoveis['preco'] <= usuario['preco_max'])]

# Criar um vetor de características para comparação
features = ['preco', 'quartos', 'garagem', 'piscina']
scaler = MinMaxScaler()
imoveis_filtrados[features] = scaler.fit_transform(imoveis_filtrados[features])

usuario_features = np.array([[usuario['preco_max'], usuario['quartos'], usuario['garagem'], usuario['piscina']]])
usuario_features = scaler.transform(usuario_features)

# Calcular similaridade entre as preferências do usuário e os imóveis disponíveis
similaridade = cosine_similarity(usuario_features, imoveis_filtrados[features])
imoveis_filtrados['score'] = similaridade[0]

# Criando um modelo preditivo de IA para recomendações futuras
X = imoveis[features]
y = np.array([feedback_historico.get(str(i), np.nan) for i in imoveis['id']])  # Usando feedback real sempre que disponível

# Remover valores NaN (caso haja imóveis sem feedback)
validos = ~np.isnan(y)
X, y = X[validos], y[validos]

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Prevendo o interesse do usuário nos imóveis recomendados
imoveis_filtrados['predict_score'] = modelo.predict(imoveis_filtrados[features])

# Ordenar as recomendações pelo score preditivo
recomendacoes = imoveis_filtrados.sort_values(by='predict_score', ascending=False)

# Função para registrar feedback e salvar no arquivo JSON
def registrar_feedback(imovel_id, nota):
    feedback_historico[str(imovel_id)] = nota
    with open(feedback_file, "w") as f:
        json.dump(feedback_historico, f)
    st.success(f"Feedback registrado: Imóvel {imovel_id} recebeu nota {nota}")

# Função para enviar notificações por email (simulação)
def enviar_notificacao(email, mensagem):
    st.info(f"Notificação enviada para {email}: {mensagem}")

# Criando interface com Streamlit
st.title("Recomendação de Imóveis")
st.write("Sistema de recomendação baseado em IA e feedback do usuário.")

for index, row in recomendacoes.iterrows():
    st.subheader(f"Imóvel {int(row['id'])} - {row['localizacao']}")
    st.write(f"Preço: {int(row['preco'])}€")
    st.write(f"Quartos: {int(row['quartos'])} | Garagem: {'Sim' if row['garagem'] else 'Não'} | Piscina: {'Sim' if row['piscina'] else 'Não'}")
    nota = st.slider(f"Dê uma nota (0-5) para o imóvel {int(row['id'])}", 0.0, 5.0, 3.0, 0.5)
    if st.button(f"Salvar Feedback para Imóvel {int(row['id'])}"):
        registrar_feedback(int(row['id']), nota)
        enviar_notificacao("user@email.com", f"Você avaliou o imóvel {int(row['id'])} com {nota} estrelas.")
