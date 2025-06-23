# dashboard.py

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuração da Página do Dashboard ---
st.set_page_config(
    page_title="Dashboard de Vendas",
    page_icon="📊",
    layout="wide"
)


# --- Carregamento e Processamento dos Dados ---
@st.cache_data
def carregar_e_processar_dados():
    """
    Carrega, une, limpa e transforma os dados de vendas,
    tratando corretamente os separadores decimais.
    """
    try:
        # Carrega os dados. Assumimos que a coluna 'genero' está em 'clientes.csv'
        df_clientes = pd.read_csv('clientes.csv', delimiter=';')
        df_vendas = pd.read_csv('vendas.csv', delimiter=';', decimal=',')
        df_produtos = pd.read_csv('produtos_vendidos.csv', delimiter=';', decimal=',')
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo .csv não encontrado. Verifique o caminho: {e.filename}")
        return None

    # Merge das tabelas
    df_merged = pd.merge(df_vendas, df_produtos, on='venda_id', how='left')
    df_full = pd.merge(df_merged, df_clientes, on='cliente_id', how='left')

    # Limpeza e Transformação
    df_full['DATA'] = pd.to_datetime(df_full['DATA'], errors='coerce')
    df_full['data_nascimento'] = pd.to_datetime(df_full['data_nascimento'], errors='coerce')

    df_full.dropna(subset=['DATA', 'data_nascimento'], inplace=True)

    df_full['idade_cliente'] = df_full['data_nascimento'].apply(
        lambda x: relativedelta(datetime.now(), x).years
    )

    numeric_cols = ['preco_promocional', 'valor_unitario', 'produto_custo', 'quantidade_vendida', 'valor_final']
    for col in numeric_cols:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0)

    df_full['preco_venda_item'] = np.where(
        df_full['preco_promocional'] == 0,
        df_full['valor_unitario'],
        df_full['preco_promocional']
    )
    df_full['lucro_item'] = (df_full['preco_venda_item'] - df_full['produto_custo']) * df_full['quantidade_vendida']
    df_full['ano_mes'] = df_full['DATA'].dt.to_period('M').astype(str)

    return df_full


# ==============================================================================
# --- NOVA FUNÇÃO PARA ANÁLISE PREDITIVA ---
# ==============================================================================
@st.cache_data
def gerar_previsao_vendas():
    """
    Carrega os dados de vendas, treina um modelo SARIMA e retorna
    um gráfico com a previsão de vendas para os próximos 12 meses.
    """
    # Carregamento e preparação dos dados dentro da função
    df_vendas_pred = pd.read_csv('vendas.csv', delimiter=';', decimal=',')
    df_vendas_pred['DATA'] = pd.to_datetime(df_vendas_pred['DATA'], errors='coerce')
    df_vendas_pred['valor_final'] = pd.to_numeric(df_vendas_pred['valor_final'], errors='coerce')
    df_vendas_pred.dropna(subset=['DATA', 'valor_final'], inplace=True)
    df_vendas_pred.set_index('DATA', inplace=True)
    ts_vendas = df_vendas_pred['valor_final'].resample('M').sum()

    # Modelagem Preditiva com SARIMA
    modelo = SARIMAX(ts_vendas, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    resultado = modelo.fit(disp=False)

    # Previsão para os próximos 12 meses
    previsao_passos = 12
    previsao_resultado = resultado.get_forecast(steps=previsao_passos)
    previsao_media = previsao_resultado.predicted_mean
    intervalo_confianca = previsao_resultado.conf_int()

    # Visualização dos Resultados
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.plot(ts_vendas.index, ts_vendas, label='Vendas Históricas', color='royalblue')
    ax.plot(previsao_media.index, previsao_media.values, label='Previsão de Vendas', color='darkorange', linestyle='--')
    ax.fill_between(intervalo_confianca.index, intervalo_confianca.iloc[:, 0], intervalo_confianca.iloc[:, 1],
                    color='sandybrown', alpha=0.3, label='Intervalo de Confiança (95%)')
    ax.set_title(f'Previsão de Vendas para os Próximos {previsao_passos} Meses', fontsize=18)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Receita (R$)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    return fig


# --- Carregamento Principal ---
df_completo = carregar_e_processar_dados()

if df_completo is None:
    st.stop()

# --- Título do Dashboard ---
st.title("📊 Análise de Vendas da Empresa Varejista")
st.markdown("---")

# --- Filtros na Barra Lateral ---
st.sidebar.header("Filtros Dinâmicos")

bairros = sorted(df_completo['nome_set'].astype(str).unique())
select_all_bairros = st.sidebar.checkbox("Selecionar Todos os Bairros", value=True)
bairro_selecionado = st.sidebar.multiselect(
    "Selecione o Bairro do Cliente",
    options=bairros,
    default=bairros if select_all_bairros else []
)

st.sidebar.markdown("---")

periodos = sorted(df_completo['ano_mes'].unique())
select_all_periodos = st.sidebar.checkbox("Selecionar Todos os Períodos", value=True)
periodo_selecionado = st.sidebar.multiselect(
    "Selecione o Período (Ano-Mês)",
    options=periodos,
    default=periodos if select_all_periodos else []
)

# Aplicar filtros
df_filtrado = df_completo[
    (df_completo['nome_set'].isin(bairro_selecionado)) &
    (df_completo['ano_mes'].isin(periodo_selecionado))
    ]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste sua seleção.")
    st.stop()

# --- Visão Geral (KPIs) ---
st.header("Visão Geral do Período Selecionado")

df_vendas_unicas = df_filtrado.drop_duplicates(subset=['venda_id'])
total_receita = df_vendas_unicas['valor_final'].sum()
total_lucro = df_filtrado['lucro_item'].sum()
num_vendas = df_vendas_unicas['venda_id'].nunique()
ticket_medio = total_receita / num_vendas if num_vendas > 0 else 0
total_clientes = df_vendas_unicas['cliente_id'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Receita Total", f"R$ {total_receita:,.2f}")
col2.metric("Lucro Total", f"R$ {total_lucro:,.2f}")
col3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
col4.metric("Clientes Atendidos", f"{total_clientes}")

st.markdown("---")

# --- Gráficos Dinâmicos ---
st.header("Análises Gráficas")
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Top 10 Bairros por Receita")
    receita_bairro = df_vendas_unicas.groupby('nome_set')['valor_final'].sum().nlargest(10).sort_values(ascending=True)
    if not receita_bairro.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=receita_bairro.values, y=receita_bairro.index, ax=ax, orient='h', palette='viridis')
        ax.set_xlabel('Receita (R$)')
        ax.set_ylabel('Bairro')
        for index, value in enumerate(receita_bairro):
            ax.text(value, index, f' R$ {value:,.0f}', va='center')
        st.pyplot(fig)
    else:
        st.info("Não há dados de receita por bairro para os filtros selecionados.")

with col_graph2:
    st.subheader("Vendas por Forma de Pagamento")
    vendas_pagamento = df_vendas_unicas['forma_pagamento'].value_counts()
    if not vendas_pagamento.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(vendas_pagamento, labels=vendas_pagamento.index, autopct='%1.1f%%', startangle=140,
               colors=sns.color_palette('pastel'))
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("Não há dados de forma de pagamento para os filtros selecionados.")

st.markdown("<br>", unsafe_allow_html=True)
col_graph3, col_graph4 = st.columns(2)

with col_graph3:
    st.subheader("Top 10 Produtos Mais Vendidos")
    top_10_produtos = df_filtrado.groupby('produto_nome')['quantidade_vendida'].sum().nlargest(10).sort_values(
        ascending=True)
    if not top_10_produtos.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_10_produtos.values, y=top_10_produtos.index, ax=ax, orient='h', palette='rocket')
        ax.set_xlabel('Quantidade Vendida')
        ax.set_ylabel('Produto')
        st.pyplot(fig)
    else:
        st.info("Não há dados de produtos vendidos para os filtros selecionados.")

with col_graph4:
    st.subheader("Top 10 Fabricantes por Lucro")
    lucro_fabricante = df_filtrado.groupby('fabricante')['lucro_item'].sum().nlargest(10).sort_values(ascending=True)
    if not lucro_fabricante.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=lucro_fabricante.values, y=lucro_fabricante.index, ax=ax, orient='h', palette='mako')
        ax.set_xlabel('Lucro (R$)')
        ax.set_ylabel('Fabricante')
        st.pyplot(fig)
    else:
        st.info("Não há dados de lucro por fabricante para os filtros selecionados.")

st.markdown("---")

st.header("Análise de Clientes")
col_genero, col_idade = st.columns(2)

with col_genero:
    st.subheader("Distribuição de Clientes por Gênero")
    contagem_genero = df_vendas_unicas['genero'].value_counts()
    if not contagem_genero.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(contagem_genero, labels=contagem_genero.index, autopct='%1.1f%%', startangle=90,
               colors=sns.color_palette('pastel'))
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("Não há dados de gênero para os filtros selecionados.")

with col_idade:
    st.subheader("Distribuição de Idade dos Clientes")
    if not df_vendas_unicas['idade_cliente'].empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_vendas_unicas['idade_cliente'], bins=20, kde=True, ax=ax, color='skyblue')
        ax.set_xlabel('Idade')
        ax.set_ylabel('Número de Clientes')
        st.pyplot(fig)
    else:
        st.info("Não há dados de idade para os filtros selecionados.")

st.markdown("---")

st.subheader("Evolução Mensal da Receita")
vendas_mensais = df_vendas_unicas.set_index('DATA').resample('M')['valor_final'].sum()
if not vendas_mensais.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vendas_mensais.index, vendas_mensais.values, marker='o', linestyle='-')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Receita (R$)')
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("Não há dados de vendas mensais para os filtros selecionados.")

# ==============================================================================
# --- SEÇÃO DE ANÁLISE PREDITIVA INTEGRADA ---
# ==============================================================================
st.markdown("---")
st.header("🔮 Análise Preditiva")

with st.expander("Clique aqui para ver a Previsão de Vendas para os Próximos 12 Meses"):
    try:
        # Gerar e exibir o gráfico de previsão a partir da função cacheada
        fig_previsao = gerar_previsao_vendas()
        st.pyplot(fig_previsao)
        st.info(
            "A previsão utiliza o histórico completo de vendas para estimar os valores futuros, independentemente dos filtros selecionados na barra lateral.")
    except Exception as e:
        st.error(f"Não foi possível gerar a previsão de vendas. Erro: {e}")

# --- SEÇÃO DE RELATÓRIO/METODOLOGIA ---
st.markdown("---")
with st.expander("ℹ️ Sobre o Projeto e Metodologia"):
    st.markdown("""
    **Objetivo:** Realizar uma análise de vendas de uma empresa varejista, construindo um dashboard interativo para extração de insights e um modelo de previsão de receita.

    **Fontes de Dados:**
    - `clientes.csv`: Informações demográficas dos clientes.
    - `vendas.csv`: Registro de transações.
    - `produtos_vendidos.csv`: Detalhes dos itens por venda.

    **Tecnologias Utilizadas:**
    - **Linguagem:** Python
    - **Bibliotecas:** Streamlit (Dashboard), Pandas (Manipulação de Dados), Matplotlib/Seaborn (Visualização), Statsmodels (Modelagem Preditiva).

    **Métricas e Análises:**
    - Os KPIs e gráficos foram desenvolvidos para fornecer uma visão clara da performance de vendas, lucratividade, comportamento do cliente e eficiência de produtos, conforme solicitado no escopo do projeto.
    """)