
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Automação-IA", layout="wide")

st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    ">
        <div style="font-size: 56px; margin-bottom: 10px;">🤖📊</div>
        <h1 style="color: #00d4ff; font-size: 2.4em; margin: 0 0 8px 0; font-weight: 800; letter-spacing: 1px;">
            Automação-IA
        </h1>
        <p style="color: #a0c4d8; font-size: 1.1em; margin: 0;">
            Automação Inteligente de Análise de Dados com Machine Learning
        </p>
        <div style="margin-top: 18px; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <span style="background: rgba(0,212,255,0.15); color: #00d4ff; padding: 6px 16px; border-radius: 20px; font-size: 0.85em;">📂 Upload de CSV</span>
            <span style="background: rgba(0,212,255,0.15); color: #00d4ff; padding: 6px 16px; border-radius: 20px; font-size: 0.85em;">📈 Visualizações</span>
            <span style="background: rgba(0,212,255,0.15); color: #00d4ff; padding: 6px 16px; border-radius: 20px; font-size: 0.85em;">🧠 Clustering IA</span>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

file_input = st.file_uploader("Faça upload do seu arquivo CSV", type="csv")

if file_input is None:
    st.info("Aguardando upload de um arquivo CSV para iniciar a análise.")
    st.stop()

try:
    df = pd.read_csv(file_input)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

if df.empty:
    st.error("O arquivo CSV está vazio.")
    st.stop()

st.success(f"Arquivo carregado com sucesso! {len(df)} registros e {len(df.columns)} colunas.")

# Resumo dos dados
with st.expander("Visualizar dados e estatísticas"):
    tab1, tab2, tab3 = st.tabs(["Primeiras linhas", "Estatísticas", "Valores nulos"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)
    with tab3:
        nulos = df.isnull().sum().reset_index()
        nulos.columns = ["Coluna", "Valores Nulos"]
        st.dataframe(nulos, use_container_width=True)

st.markdown("---")

# Seção de gráficos
st.subheader("Gerador de Gráficos")
col1, col2 = st.columns(2)

with col1:
    variavel = st.selectbox("Escolha a variável para análise:", df.columns)

with col2:
    tipo_grafico = st.selectbox(
        "Escolha o tipo de gráfico:",
        ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Boxplot", "Bubble Chart"]
    )

if st.button("Gerar Gráfico", use_container_width=True):
    colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    outra_coluna = next((c for c in df.columns if c != variavel), None)

    try:
        if tipo_grafico == "Scatter Plot":
            if outra_coluna:
                fig = px.scatter(df, x=variavel, y=outra_coluna, title=f"Scatter: {variavel} vs {outra_coluna}")
            else:
                st.warning("Precisa de pelo menos 2 colunas para Scatter Plot.")
                st.stop()
        elif tipo_grafico == "Line Plot":
            if outra_coluna:
                fig = px.line(df, x=variavel, y=outra_coluna, title=f"Linha: {variavel} vs {outra_coluna}")
            else:
                st.warning("Precisa de pelo menos 2 colunas para Line Plot.")
                st.stop()
        elif tipo_grafico == "Bar Chart":
            if outra_coluna:
                fig = px.bar(df, x=variavel, y=outra_coluna, title=f"Barras: {variavel} vs {outra_coluna}")
            else:
                st.warning("Precisa de pelo menos 2 colunas para Bar Chart.")
                st.stop()
        elif tipo_grafico == "Histogram":
            fig = px.histogram(df, x=variavel, title=f"Histograma: {variavel}")
        elif tipo_grafico == "Boxplot":
            fig = px.box(df, y=variavel, title=f"Boxplot: {variavel}")
        elif tipo_grafico == "Bubble Chart":
            if len(df.columns) >= 3:
                tamanho_col = [c for c in colunas_numericas if c != variavel and c != df.columns[0]]
                if tamanho_col:
                    fig = px.scatter(df, x=df.columns[0], y=variavel, size=tamanho_col[0],
                                     size_max=60, title=f"Bubble: {variavel}")
                else:
                    st.warning("Sem coluna numérica suficiente para o tamanho das bolhas.")
                    st.stop()
            else:
                st.warning("Precisa de pelo menos 3 colunas para o Bubble Chart.")
                st.stop()

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")

st.markdown("---")

# Seção de Clustering
st.subheader("Clustering com IA (KMeans)")

colunas_num = df.select_dtypes(include=np.number).columns.tolist()

if not colunas_num:
    st.warning("Não há colunas numéricas no dataset para realizar o clustering.")
else:
    col_cluster = st.selectbox("Escolha a coluna para o clustering:", colunas_num,
                                index=colunas_num.index("renda") if "renda" in colunas_num else 0)
    n_clusters = st.slider("Número de clusters:", min_value=2, max_value=10, value=5)

    if st.button("Gerar Cluster com IA", use_container_width=True):
        dados_cluster = df[[col_cluster]].dropna()

        if len(dados_cluster) < n_clusters:
            st.error(f"Dados insuficientes para {n_clusters} clusters.")
        else:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(dados_cluster)
                df_cluster = df.loc[dados_cluster.index].copy()
                df_cluster["Cluster"] = kmeans.labels_.astype(str)

                fig = px.box(df_cluster, x="Cluster", y=col_cluster, color="Cluster", notched=True,
                             title=f"Distribuição de '{col_cluster}' por Cluster")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Centróides dos clusters:**")
                centroides = pd.DataFrame(kmeans.cluster_centers_, columns=[col_cluster])
                centroides.index.name = "Cluster"
                centroides[col_cluster] = centroides[col_cluster].round(2)
                st.dataframe(centroides, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao gerar clusters: {e}")
