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
        # Ajustado para buscar da subpasta 'atividadeFinal', caso necessário.
        # Se os arquivos estiverem na mesma pasta do script, remova 'atividadeFinal/'.
        df_clientes = pd.read_csv('clientes.csv', delimiter=';')
        df_vendas = pd.read_csv('vendas.csv', delimiter=';', decimal=',')
        df_produtos = pd.read_csv('produtos_vendidos.csv', delimiter=';', decimal=',')
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo .csv não encontrado. Verifique o caminho: {e.filename}")
        return None

    df_merged = pd.merge(df_vendas, df_produtos, on='venda_id', how='left')
    df_full = pd.merge(df_merged, df_clientes, on='cliente_id', how='left')

    df_full['DATA'] = pd.to_datetime(df_full['DATA'], errors='coerce')
    df_full['data_nascimento'] = pd.to_datetime(df_full['data_nascimento'], errors='coerce')
    df_full.dropna(subset=['DATA', 'data_nascimento', 'nome_set'], inplace=True)

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


@st.cache_data
def gerar_previsao_vendas():
    """
    Carrega os dados de vendas, treina um modelo SARIMA e retorna
    um gráfico com a previsão de vendas para os próximos 12 meses.
    """
    df_vendas_pred = pd.read_csv('vendas.csv', delimiter=';', decimal=',')
    df_vendas_pred['DATA'] = pd.to_datetime(df_vendas_pred['DATA'], errors='coerce')
    df_vendas_pred['valor_final'] = pd.to_numeric(df_vendas_pred['valor_final'], errors='coerce')
    df_vendas_pred.dropna(subset=['DATA', 'valor_final'], inplace=True)
    df_vendas_pred.set_index('DATA', inplace=True)
    ts_vendas = df_vendas_pred['valor_final'].resample('M').sum()

    modelo = SARIMAX(ts_vendas, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    resultado = modelo.fit(disp=False)

    previsao_passos = 12
    previsao_resultado = resultado.get_forecast(steps=previsao_passos)
    previsao_media = previsao_resultado.predicted_mean
    intervalo_confianca = previsao_resultado.conf_int()

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
    # CORREÇÃO: Usar a coluna 'Gênero' com 'G' maiúsculo
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

# ==============================================================================
# --- SEÇÃO DE EVOLUÇÃO DA RECEITA COM SELETORES (ATENDENDO AO PEDIDO) ---
# ==============================================================================
st.markdown("---")
st.header("📈 Evolução da Receita")

# Adicionando os botões de rádio para seleção de período
periodo_view = st.radio(
    "Selecione a granularidade do período para visualizar a evolução:",
    ('Mensal', 'Trimestral', 'Semanal'),
    horizontal=True,
    key='periodo_evolucao_radio'
)

# Mapear a seleção para a frequência do Pandas
freq_map = {'Mensal': 'M', 'Trimestral': 'Q', 'Semanal': 'W'}
resample_freq = freq_map[periodo_view]

st.subheader(f"Visão {periodo_view} da Receita (Período Filtrado)")

# Resample e plot
vendas_periodo = df_vendas_unicas.set_index('DATA').resample(resample_freq)['valor_final'].sum()
if not vendas_periodo.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(vendas_periodo.index, vendas_periodo.values, marker='o', linestyle='-')
    ax.set_xlabel(f'Período ({periodo_view})')
    ax.set_ylabel('Receita (R$)')
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info(f"Não há dados de vendas para a visão {periodo_view} com os filtros selecionados.")


# ==============================================================================
# --- ANÁLISE SAZONAL ---
# ==============================================================================
st.markdown("---")
st.header("📅 Análise Sazonal")
st.subheader("Receita Total por Mês (Todos os Anos)")

df_sazonal = df_completo.copy()
df_sazonal['mes_numero'] = df_sazonal['DATA'].dt.month
receita_por_mes = df_sazonal.groupby('mes_numero')['valor_final'].sum()

nomes_meses = {
    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
}
receita_por_mes.index = receita_por_mes.index.map(nomes_meses)
ordem_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
receita_por_mes = receita_por_mes.reindex(ordem_meses).fillna(0)

if not receita_por_mes.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=receita_por_mes.index, y=receita_por_mes.values, ax=ax, palette='plasma')
    ax.set_xlabel('Mês do Ano')
    ax.set_ylabel('Receita Total Acumulada (R$)')
    ax.set_title('Sazonalidade de Vendas: Receita Total por Mês', fontsize=12)
    st.pyplot(fig)
    st.info("Este gráfico mostra a soma da receita de cada mês, acumulada durante todo o período histórico dos dados.")
else:
    st.info("Não há dados suficientes para analisar a sazonalidade.")

# ==============================================================================
# --- ANÁLISE PREDITIVA ---
# ==============================================================================
st.markdown("---")
st.header("🔮 Análise Preditiva")

with st.expander("Clique aqui para ver a Previsão de Vendas para os Próximos 12 Meses"):
    try:
        fig_previsao = gerar_previsao_vendas()
        st.pyplot(fig_previsao)
        st.info(
            "A previsão utiliza o histórico completo de vendas para estimar os valores futuros, independentemente dos filtros selecionados na barra lateral.")
    except Exception as e:
        st.error(f"Não foi possível gerar a previsão de vendas. Erro: {e}")
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