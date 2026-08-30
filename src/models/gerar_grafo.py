import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import networkx as nx
import numpy as np

PATH = 'reports/matriz_importancia_2026-08-28_09-52-02.csv'

df = pd.read_csv(PATH, index_col = 'Unnamed: 0')

matriz_numpy = df.to_numpy()

matriz_min = np.minimum(matriz_numpy,matriz_numpy.T)

df = pd.DataFrame(
    matriz_min, index= df.index, columns= df.columns
)

print(df)
def plotar_grafo(matriz, IMPORT_MIN):
    G = nx.from_pandas_adjacency(matriz, create_using=nx.Graph)
    
    arestas_para_remover = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["weight"] <= IMPORT_MIN
    ]
    G.remove_edges_from(arestas_para_remover)
    
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # --- ESTILIZAÇÃO DO GRAFO ---
    plt.figure(figsize=(10,15), dpi=200)
    
    pos = nx.spring_layout(G, weight='weight', k = 1.2, iterations=100, seed=1)
    
    pesos = np.array([d["weight"] for u, v, d in G.edges(data=True)])
    
    nx.draw_networkx_nodes(
        G, pos, node_size=700, node_color="skyblue", edgecolors="black"
    )
    
    if len(pesos) > 0:
        espessura = pesos/pesos.max()
    else: 
        espessura = 5
    nx.draw_networkx_edges(
        G, pos, width=espessura, edge_color="gray", alpha=0.7, arrows=False
    )
    
    
    # Desenha os Nomes das Variáveis
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    
    plt.title(
        f"Grafo de Importância Mínima entre Variáveis (Limiar > {IMPORT_MIN})",
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'reports/grafo{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')

plotar_grafo(df, 0.8)