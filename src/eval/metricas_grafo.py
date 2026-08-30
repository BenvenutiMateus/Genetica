import os
import pandas as pd
import networkx as nx
import numpy as np
from src.utils.transformacoes import imp_to_adj

df = pd.read_csv('reports/matriz_importancia_2026-08-29_01-23-51.csv', index_col = 0)

df = imp_to_adj(df)

G = nx.from_pandas_adjacency(df)
# 1. Calculo das principais medidas
grau_absoluto = dict(G.degree())
centralidade_grau = nx.degree_centrality(G)
intermediacao = nx.betweenness_centrality(G)
proximidade = nx.closeness_centrality(G)
autovetor = nx.eigenvector_centrality(G, max_iter=1000)

# 2. Consolidacao em um DataFrame do Pandas
df_metricas = pd.DataFrame({
    'Degree': pd.Series(grau_absoluto),
    'Degree centrality': pd.Series(centralidade_grau),
    'Betweenneess': pd.Series(intermediacao),
    'Closeness': pd.Series(proximidade),
    'Eigen Centrality': pd.Series(autovetor)
})

df_metricas = df_metricas.sort_values(by=['Betweenneess'], ascending=False)

print(df_metricas.head())


caminho_saida = 'reports/metricas_completas_do_grafo.csv'
df_metricas.to_csv(caminho_saida)

print(f"Sucesso! Arquivo exportado em: {caminho_saida}")