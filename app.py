import os
import time
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from src.utils.transformacoes import imp_to_adj


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA = "models/Importancias_2026-09-05_22-04-33"
PASTA_IMPORTANCIAS = os.path.join(PASTA, "importancias")


# ============================================================
# FUNÇÃO PARA DESCOBRIR ALVOS E VARIÁVEIS
# ============================================================

@st.cache_data
def carregar_dados():
    arquivos = [
        f for f in os.listdir(PASTA_IMPORTANCIAS)
        if f.endswith(".csv")
    ]

    alvos = []
    todas_colunas = set()

    for arquivo in arquivos:
        target = arquivo.replace(".csv", "")
        alvos.append(target)

        caminho = os.path.join(PASTA_IMPORTANCIAS, arquivo)
        dados = pd.read_csv(caminho)

        todas_colunas.update(dados["variavel"].unique())

    return sorted(alvos), sorted(todas_colunas)


# ============================================================
# MATRIZ DE ESTABILIDADE
# ============================================================

@st.cache_data
def calcular_matriz_estabilidade(p, q, iteracoes=None):

    alvos, todas_colunas = carregar_dados()

    matriz_estabilidade = pd.DataFrame(
        0.0,
        index=alvos,
        columns=todas_colunas
    )

    for target in alvos:

        arquivo = os.path.join(
            PASTA_IMPORTANCIAS,
            f"{target}.csv"
        )

        if not os.path.exists(arquivo):
            continue

        dados = pd.read_csv(arquivo)

        iteracoes_disponiveis = sorted(
            dados["iteracao"].unique()
        )

        if iteracoes is not None:
            iteracoes_disponiveis = (
                iteracoes_disponiveis[:iteracoes]
            )

        n_usadas = 0

        for i in iteracoes_disponiveis:

            grupo = (
                dados[dados["iteracao"] == i]
                .set_index("variavel")["importancia_bruta"]
            )

            grupo_ordenado = grupo.sort_values(
                ascending=False
            )

            soma_total = grupo_ordenado.sum()

            if soma_total > 0:
                importancias_normalizadas = (
                    grupo_ordenado / soma_total
                )
            else:
                importancias_normalizadas = grupo_ordenado

            soma_cumulativa = (
                importancias_normalizadas.cumsum()
            )

            variaveis_selecionadas = (
                soma_cumulativa[
                    soma_cumulativa.shift(
                        fill_value=0
                    ) < p
                ].index
            )

            # Garante pelo menos q variáveis
            if len(variaveis_selecionadas) < q:
                variaveis_selecionadas = (
                    soma_cumulativa.index[:q]
                )

            matriz_estabilidade.loc[
                target,
                variaveis_selecionadas
            ] += 1.0

            n_usadas += 1

        if n_usadas > 0:
            matriz_estabilidade.loc[target] /= n_usadas

    return matriz_estabilidade


# ============================================================
# FUNÇÃO DO GRAFO
# ============================================================

def plotar_grafo(matriz, import_min):

    # Transforma a matriz de importância em matriz de adjacência
    matriz_adj = imp_to_adj(matriz)

    G = nx.from_pandas_adjacency(
        matriz_adj,
        create_using=nx.Graph
    )

    # --------------------------------------------------------
    # REMOVE ARESTAS ABAIXO DO LIMIAR
    # --------------------------------------------------------

    arestas_para_remover = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["weight"] <= import_min
    ]

    G.remove_edges_from(arestas_para_remover)

    # Remove nós isolados
    G.remove_nodes_from(
        list(nx.isolates(G))
    )

    # --------------------------------------------------------
    # FIGURA
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 15),
        dpi=150
    )

    if len(G.nodes) == 0:

        ax.text(
            0.5,
            0.5,
            "Nenhuma conexão acima do IMPORT_MIN",
            ha="center",
            va="center",
            fontsize=16
        )

        ax.axis("off")

        return fig, G

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    pos = nx.spring_layout(
        G,
        weight="weight",
        k=2,
        iterations=200,
        seed=1
    )

    # --------------------------------------------------------
    # PESOS
    # --------------------------------------------------------

    pesos = np.array([
        d["weight"]
        for _, _, d in G.edges(data=True)
    ])

    if len(pesos) > 0 and pesos.max() > 0:
        espessuras = 2 * (
            pesos / pesos.max()
        )
    else:
        espessuras = 1

    # --------------------------------------------------------
    # NÓS
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=700,
        node_color="skyblue",
        edgecolors="black",
        ax=ax
    )

    # --------------------------------------------------------
    # ARESTAS
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        width=espessuras,
        edge_color="gray",
        alpha=0.7,
        ax=ax
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    ax.set_title(
        f"Grafo de Importância\n"
        f"Limiar > {import_min:.3f}",
        fontsize=16
    )

    ax.axis("off")
    fig.tight_layout()

    return fig, G


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Explorador de Estabilidade",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Explorador de Estabilidade")
st.markdown(
    "Ajuste **q**, **p** e **IMPORT_MIN** para explorar "
    "a estrutura do grafo."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Parâmetros")

q = st.sidebar.slider(
    "q — mínimo de variáveis",
    min_value=1,
    max_value=50,
    value=3,
    step=1
)

p = st.sidebar.slider(
    "p — proporção acumulada",
    min_value=0.0,
    max_value=1.00,
    value=0.80,
    step=0.01
)

import_min = st.sidebar.slider(
    "IMPORT_MIN — limiar do grafo",
    min_value=0.00,
    max_value=1.00,
    value=0.90,
    step=0.01
)


# ============================================================
# MOSTRA PARÂMETROS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "q",
        q
    )

with col2:
    st.metric(
        "p",
        f"{p:.2f}"
    )

with col3:
    st.metric(
        "IMPORT_MIN",
        f"{import_min:.2f}"
    )


# ============================================================
# CALCULA MATRIZ
# ============================================================

with st.spinner("Calculando matriz de estabilidade..."):

    inicio = time.perf_counter()

    matriz_estabilidade = calcular_matriz_estabilidade(
        p=p,
        q=q
    )

    tempo = time.perf_counter() - inicio


st.caption(
    f"Matriz calculada em {tempo:.3f} segundos"
)


# ============================================================
# GRAFO
# ============================================================

fig, G = plotar_grafo(
    matriz_estabilidade,
    import_min
)

st.pyplot(
    fig,
    use_container_width=True
)


# ============================================================
# ESTATÍSTICAS
# ============================================================

st.subheader("📊 Informações do grafo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Nós",
        G.number_of_nodes()
    )

with col2:
    st.metric(
        "Arestas",
        G.number_of_edges()
    )

with col3:
    if G.number_of_nodes() > 0:
        densidade = nx.density(G)
    else:
        densidade = 0

    st.metric(
        "Densidade",
        f"{densidade:.4f}"
    )


# ============================================================
# MATRIZ
# ============================================================

with st.expander("Ver matriz de estabilidade"):

    st.dataframe(
        matriz_estabilidade.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )