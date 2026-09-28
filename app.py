import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.transformacoes import imp_to_adj


# ============================================================
# CONFIGURACAO
# ============================================================

# ============================================================
# CARREGAR ALVOS E VARIAVEIS
# ============================================================

@st.cache_data
def carregar_dados(pasta_importancias):

    if not os.path.exists(pasta_importancias):
        return [], []

    arquivos = [
        f
        for f in os.listdir(pasta_importancias)
        if f.endswith(".csv")
    ]

    alvos = []
    todas_colunas = set()

    for arquivo in arquivos:

        target = arquivo.replace(
            ".csv",
            ""
        )

        alvos.append(target)

        caminho = os.path.join(
            pasta_importancias,
            arquivo
        )

        dados = pd.read_csv(
            caminho
        )

        todas_colunas.update(
            dados["variavel"].unique()
        )

    return (
        sorted(alvos),
        sorted(todas_colunas)
    )


# ============================================================
# CALCULAR MATRIZ DE ESTABILIDADE
# ============================================================

@st.cache_data
def calcular_matriz_estabilidade(
    pasta_importancias,
    p,
    q,
    iteracoes=None
):

    alvos, todas_colunas = carregar_dados(
        pasta_importancias
    )

    if not alvos:
        return pd.DataFrame()

    matriz_estabilidade = pd.DataFrame(
        0.0,
        index=alvos,
        columns=todas_colunas
    )

    for target in alvos:

        arquivo = os.path.join(
            pasta_importancias,
            f"{target}.csv"
        )

        if not os.path.exists(arquivo):
            continue

        dados = pd.read_csv(
            arquivo
        )

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
                dados[
                    dados["iteracao"] == i
                ]
                .set_index("variavel")[
                    "importancia_bruta"
                ]
            )

            # ==================================================
            # ORDENAR IMPORTANCIAS
            # ==================================================

            grupo_ordenado = (
                grupo.sort_values(
                    ascending=False
                )
            )

            # ==================================================
            # NORMALIZAR
            # ==================================================

            soma_total = (
                grupo_ordenado.sum()
            )

            if soma_total > 0:

                importancias_normalizadas = (
                    grupo_ordenado
                    / soma_total
                )

            else:

                importancias_normalizadas = (
                    grupo_ordenado
                )

            # ==================================================
            # SOMA CUMULATIVA
            # ==================================================

            soma_cumulativa = (
                importancias_normalizadas
                .cumsum()
            )

            # ==================================================
            # CRITERIO p
            #
            # p = 0:
            # nenhuma entra pelo critério p
            # ==================================================

            variaveis_selecionadas = (
                soma_cumulativa[
                    soma_cumulativa.shift(
                        fill_value=0
                    ) < p
                ].index
            )

            # ==================================================
            # CRITERIO q
            #
            # garante pelo menos q variáveis
            # ==================================================

            if len(
                variaveis_selecionadas
            ) < q:

                variaveis_selecionadas = (
                    grupo_ordenado
                    .index[:q]
                )

            # ==================================================
            # FREQUENCIA DE SELECAO
            # ==================================================

            matriz_estabilidade.loc[
                target,
                variaveis_selecionadas
            ] += 1.0

            n_usadas += 1

        # ======================================================
        # TRANSFORMA EM PROPORCAO
        # ======================================================

        if n_usadas > 0:

            matriz_estabilidade.loc[
                target
            ] /= n_usadas

    return matriz_estabilidade


def calcular_limite_esperado_v(
    q,
    numero_variaveis,
    limiar
):
    """Calcula o limite superior de E[V] para o critério somente q."""

    if (
        limiar <= 0.5
        or numero_variaveis < 2
    ):
        return None

    numero_arestas_possiveis = (
        numero_variaveis
        * (numero_variaveis - 1)
        / 2
    )

    return (
        q ** 2
        / (
            (2 * limiar - 1)
            * numero_arestas_possiveis
        )
    )


# ============================================================
# GRAFO DIRECIONADO
# ============================================================

def criar_grafo_direcionado(
    matriz,
    pithr
):

    G = nx.DiGraph()

    # Todos os nós possíveis
    nos = set(
        matriz.index
    ).union(
        set(matriz.columns)
    )

    G.add_nodes_from(
        nos
    )

    # --------------------------------------------------------
    # matriz.loc[target, variavel]
    #
    # significa:
    #
    # variavel ---> target
    #
    # --------------------------------------------------------

    for target in matriz.index:

        for variavel in matriz.columns:

            # Evitar auto-laços
            if target == variavel:
                continue

            peso = matriz.loc[
                target,
                variavel
            ]

            if (
                pd.notna(peso)
                and peso > pithr
            ):

                G.add_edge(
                    variavel,
                    target,
                    weight=float(peso)
                )

    return G


# ============================================================
# GRAFO NAO DIRECIONADO
# ============================================================

def criar_grafo_nao_direcionado(
    matriz,
    pithr
):

    matriz_adj = imp_to_adj(
        matriz
    )

    G = nx.from_pandas_adjacency(
        matriz_adj,
        create_using=nx.Graph
    )

    arestas_para_remover = [
        (u, v)
        for u, v, d
        in G.edges(data=True)
        if d["weight"] <= pithr
    ]

    G.remove_edges_from(
        arestas_para_remover
    )

    return G


# ============================================================
# PLOTAR GRAFO
# ============================================================

def plotar_grafo(
    matriz,
    pithr,
    manter_isolados=False,
    direcionado=True
):

    # ========================================================
    # CRIAR O GRAFO
    # ========================================================

    if direcionado:

        G = criar_grafo_direcionado(
            matriz,
            pithr
        )

    else:

        G = criar_grafo_nao_direcionado(
            matriz,
            pithr
        )

    # ========================================================
    # REMOVER ISOLADOS
    # ========================================================

    if not manter_isolados:

        G.remove_nodes_from(
            list(
                nx.isolates(G)
            )
        )

    # ========================================================
    # FIGURA
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(15, 15),
        dpi=150
    )

    # ========================================================
    # GRAFO VAZIO
    # ========================================================

    if G.number_of_nodes() == 0:

        ax.text(
            0.5,
            0.5,
            "Nenhuma conexão acima do pithr",
            ha="center",
            va="center",
            fontsize=16
        )

        ax.axis(
            "off"
        )

        return fig, G

    # ========================================================
    # POSICAO DOS NOS
    # ========================================================

    pos = nx.spring_layout(
        G,
        weight="weight",
        k=2,
        iterations=200,
        seed=1
    )

    # ========================================================
    # PESOS
    # ========================================================

    pesos = np.array(
        [
            d["weight"]
            for _, _, d
            in G.edges(
                data=True
            )
        ]
    )

    # ========================================================
    # ESPESSURA DAS ARESTAS
    # ========================================================

    if (
        len(pesos) > 0
        and pesos.max() > 0
    ):

        espessuras = (
            1
            + 3
            * (
                pesos
                / pesos.max()
            )
        )

    else:

        espessuras = 1

    # ========================================================
    # NOS
    # ========================================================

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="skyblue",
        edgecolors="black",
        linewidths=1,
        ax=ax
    )

    # ========================================================
    # ARESTAS
    # ========================================================

    if direcionado:

        nx.draw_networkx_edges(
            G,
            pos,
            width=espessuras,
            edge_color="gray",
            alpha=0.7,

            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",

            # curva ajuda bastante quando temos
            # A -> B e B -> A
            connectionstyle="arc3,rad=0.08",

            node_size=900,
            ax=ax
        )

    else:

        nx.draw_networkx_edges(
            G,
            pos,
            width=espessuras,
            edge_color="gray",
            alpha=0.7,
            ax=ax
        )

    # ========================================================
    # LABELS
    # ========================================================

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # ========================================================
    # TITULO
    # ========================================================

    if direcionado:

        titulo = (
            "Grafo Direcionado de Estabilidade\n"
            f"Limiar > {pithr:.3f}"
        )

    else:

        titulo = (
            "Grafo Não Direcionado de Estabilidade\n"
            f"Limiar > {pithr:.3f}"
        )

    ax.set_title(
        titulo,
        fontsize=16
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    return fig, G


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title=(
        "Explorador de Estabilidade"
    ),
    page_icon="🕸️",
    layout="wide"
)

st.title(
    "🕸️ Explorador de Estabilidade"
)

st.markdown(
    "Explore a matriz de estabilidade utilizando "
    "**q**, **p**, diferentes limiares e grafos "
    "**direcionados ou não direcionados**."
)


# ============================================================
# SIDEBAR - FONTE DE DADOS
# ============================================================

st.sidebar.header(
    "Fonte de Dados"
)

caminho_base = st.sidebar.text_input(
    "Digite o caminho da pasta do experimento",
    value="",
    placeholder=(
        "Ex.: /dados/Importancias_2026-09-05_22-04-33"
    )
)


pasta_importancias_selecionada = (
    os.path.join(
        caminho_base,
        "importancias"
    )
)


# ============================================================
# VERIFICAR PASTA
# ============================================================

if not os.path.exists(
    pasta_importancias_selecionada
):

    st.error(
        f"A pasta "
        f"'{pasta_importancias_selecionada}' "
        f"não foi encontrada."
    )

    st.stop()


# ============================================================
# DESCOBRIR NUMERO DE VARIAVEIS
# ============================================================

alvos, todas_colunas = carregar_dados(
    pasta_importancias_selecionada
)

numero_nos = len(
    todas_colunas
)


if numero_nos == 0:

    st.error(
        "Nenhuma variável foi encontrada "
        "nos arquivos de importâncias."
    )

    st.stop()


# ============================================================
# METODO DE SELECAO
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.header(
    "Seleção das Variáveis"
)

metodo_selecao = (
    st.sidebar.selectbox(
        "Critério",
        options=[
            "Somente q",
            "Somente p",
            "q + p"
        ],
        index=2
    )
)


# ============================================================
# PARAMETROS
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.header(
    "Parâmetros"
)


# ============================================================
# SOMENTE q
# ============================================================

if metodo_selecao == "Somente q":

    q = st.sidebar.slider(
        "q — número de variáveis",
        min_value=1,
        max_value=numero_nos,
        value=min(
            3,
            numero_nos
        ),
        step=1
    )

    # p = 0 desliga o p
    p = 0.0


# ============================================================
# SOMENTE p
# ============================================================

elif metodo_selecao == "Somente p":

    p = st.sidebar.number_input(
        "p — proporção acumulada",
        min_value=0.001,
        max_value=1.000,
        value=0.985,
        step=0.001,
        format="%.3f"
    )

    # q = 1 deixa o p dominar
    q = 1


# ============================================================
# q + p
# ============================================================

else:

    q = st.sidebar.slider(
        "q — mínimo de variáveis",
        min_value=1,
        max_value=numero_nos,
        value=min(
            3,
            numero_nos
        ),
        step=1
    )

    p = st.sidebar.number_input(
        "p — proporção acumulada",
        min_value=0.001,
        max_value=1.000,
        value=0.985,
        step=0.001,
        format="%.3f"
    )


# ============================================================
# LIMIAR
# ============================================================

pithr = (
    st.sidebar.slider(
        "pithr — limiar do grafo",
        min_value=0.000,
        max_value=1.000,
        value=0.900,
        step=0.001,
        format="%.3f"
    )
)


# ============================================================
# TIPO DE GRAFO
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.header(
    "Grafo"
)

tipo_grafo = (
    st.sidebar.selectbox(
        "Tipo de grafo",
        options=[
            "Direcionado",
            "Não direcionado"
        ],
        index=0
    )
)


# ============================================================
# NOS ISOLADOS
# ============================================================

manter_isolados = (
    st.sidebar.checkbox(
        "Manter nós isolados?"
    )
)


# ============================================================
# INFORMACOES DO EXPERIMENTO
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "Informações do experimento"
)

st.sidebar.write(
    f"**Número de variáveis:** "
    f"{numero_nos}"
)

st.sidebar.write(
    f"**Número de alvos:** "
    f"{len(alvos)}"
)


# ============================================================
# CONFIGURACAO ATUAL
# ============================================================

st.subheader(
    "⚙️ Configuração"
)


if metodo_selecao == "Somente q":

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    with c1:

        st.metric(
            "Método",
            "Somente q"
        )

    with c2:

        st.metric(
            "q",
            q
        )

    with c3:

        st.metric(
            "pithr",
            f"{pithr:.3f}"
        )

    with c4:

        limite_ev = calcular_limite_esperado_v(
            q=q,
            numero_variaveis=numero_nos,
            limiar=pithr
        )

        st.metric(
            "E[V]",
            (
                f"{limite_ev:.4f}"
                if limite_ev is not None
                else "N/A"
            ),
            help=(
                "Limite superior de E[V] pela fórmula "
                "q² / ((2·pithr − 1) · n·(n−1)/2). "
                "A fórmula exige pithr > 0,5."
            )
        )

    with c5:

        st.metric(
            "Grafo",
            tipo_grafo
        )


elif metodo_selecao == "Somente p":

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Método",
            "Somente p"
        )

    with c2:

        st.metric(
            "p",
            f"{p:.3f}"
        )

    with c3:

        st.metric(
            "pithr",
            f"{pithr:.3f}"
        )

    with c4:

        st.metric(
            "Grafo",
            tipo_grafo
        )


else:

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    with c1:

        st.metric(
            "Método",
            "q + p"
        )

    with c2:

        st.metric(
            "q",
            q
        )

    with c3:

        st.metric(
            "p",
            f"{p:.3f}"
        )

    with c4:

        st.metric(
            "pithr",
            f"{pithr:.3f}"
        )

    with c5:

        st.metric(
            "Grafo",
            tipo_grafo
        )


# ============================================================
# CALCULAR MATRIZ
# ============================================================

with st.spinner(
    "Calculando matriz de estabilidade..."
):

    inicio = (
        time.perf_counter()
    )

    matriz_estabilidade = (
        calcular_matriz_estabilidade(
            pasta_importancias=(
                pasta_importancias_selecionada
            ),
            p=p,
            q=q
        )
    )

    tempo = (
        time.perf_counter()
        - inicio
    )


st.caption(
    f"Matriz calculada em "
    f"{tempo:.3f} segundos"
)


# ============================================================
# GRAFO
# ============================================================

direcionado = (
    tipo_grafo
    == "Direcionado"
)

fig, G = plotar_grafo(
    matriz=matriz_estabilidade,
    pithr=pithr,
    manter_isolados=manter_isolados,
    direcionado=direcionado
)

st.pyplot(
    fig,
    use_container_width=True
)


# ============================================================
# ESTATISTICAS DO GRAFO
# ============================================================

st.subheader(
    "📊 Informações do grafo"
)

c1, c2, c3 = (
    st.columns(3)
)


with c1:

    st.metric(
        "Nós",
        G.number_of_nodes()
    )


with c2:

    st.metric(
        "Arestas",
        G.number_of_edges()
    )


with c3:

    densidade = (
        nx.density(G)
        if G.number_of_nodes() > 0
        else 0
    )

    st.metric(
        "Densidade",
        f"{densidade:.4f}"
    )


# ============================================================
# INFORMACOES ESPECIFICAS DO GRAFO DIRECIONADO
# ============================================================

if direcionado and G.number_of_nodes() > 0:

    st.subheader(
        "➡️ Informações direcionais"
    )

    c1, c2 = (
        st.columns(2)
    )

    graus_saida = dict(
        G.out_degree()
    )

    graus_entrada = dict(
        G.in_degree()
    )

    with c1:

        if graus_saida:

            maior_saida = max(
                graus_saida,
                key=graus_saida.get
            )

            st.metric(
                "Maior grau de saída",
                f"{maior_saida}: "
                f"{graus_saida[maior_saida]}"
            )

    with c2:

        if graus_entrada:

            maior_entrada = max(
                graus_entrada,
                key=graus_entrada.get
            )

            st.metric(
                "Maior grau de entrada",
                f"{maior_entrada}: "
                f"{graus_entrada[maior_entrada]}"
            )


# ============================================================
# MATRIZ DE ESTABILIDADE
# ============================================================

with st.expander(
    "Ver matriz de estabilidade"
):

    st.dataframe(
        matriz_estabilidade
        .style
        .format(
            "{:.3f}"
        ),
        use_container_width=True
    )


# ============================================================
# LISTA DE ARESTAS
# ============================================================

with st.expander(
    "Ver arestas do grafo"
):

    if G.number_of_edges() == 0:

        st.info(
            "Nenhuma aresta encontrada "
            "com o limiar atual."
        )

    else:

        arestas = []

        for origem, destino, dados_aresta in G.edges(
            data=True
        ):

            arestas.append(
                {
                    "origem": origem,
                    "destino": destino,
                    "peso": dados_aresta[
                        "weight"
                    ]
                }
            )

        df_arestas = pd.DataFrame(
            arestas
        )

        df_arestas = (
            df_arestas
            .sort_values(
                "peso",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        st.dataframe(
            df_arestas.style.format(
                {
                    "peso": "{:.3f}"
                }
            ),
            use_container_width=True
        )