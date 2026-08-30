import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
import datetime
import time

df = pd.read_csv('data/limpos/banco_fenotipos_conformal.csv')

df = df.drop(columns = ['id', 'samplefilename'], errors = 'ignore')

def extrair_frequencia_estabilidade_grafo(
    df, alvos, num_trees=500, iteracoes=50, fracao_amostra=0.5, q= 0.8
):
    """Extrai a frequência de estabilidade das arestas baseado no artigo

    Fellinghauer et al. (2013).

    Parâmetros:
    - q: Número de top variáveis com maior importância que serão selecionadas
         em cada iteração (frequência binária).
    """
    # Matriz inicializada com zeros
    inicio = time.perf_counter()
    matriz_estabilidade = pd.DataFrame(
        index=alvos, columns=df.columns, dtype=np.float64
    ).fillna(0.0)

    for target in alvos:
        print(f"Rodando a variável {target}")
        colunas_categoricas = [coluna for coluna in df.columns if df[coluna].nunique() < 6]
        df_alvo = df.dropna(subset=[target])

        for i in range(iteracoes):
            if fracao_amostra > 1:
                df_sub = df_alvo.sample(n=fracao_amostra, random_state=i)
            else:
                df_sub = df_alvo.sample(frac=fracao_amostra, random_state=i)


            X = df_sub.drop(columns=[target])
            y = df_sub[target]

            if target not in colunas_categoricas:
                modelorf = RandomForestRegressor(
                    n_estimators=num_trees,
                    max_features=0.3,
                    n_jobs = -1,
                    bootstrap=False,
                )
            else:
                modelorf = RandomForestClassifier(
                    n_estimators=num_trees,
                    max_features=0.3,
                    class_weight='balanced',
                    bootstrap=False,
                    n_jobs = -1,
                )

            modelorf.fit(X, y)

            resultado_permutacao = permutation_importance(
                modelorf, X, y, n_repeats=5, random_state=i, n_jobs=-1
            )
            importancias = resultado_permutacao.importances_mean

            importancias_series = pd.Series(importancias, index=X.columns).clip(lower = 0)

            importancias_ordenadas = importancias_series.sort_values(ascending = False)
            
            # 3. Normalizar
            soma_total = importancias_ordenadas.sum()
            
            if soma_total > 0:
                importancias_normalizadas = importancias_ordenadas / soma_total
            else:
                importancias_normalizadas = importancias_ordenadas

            # 4. Calcular soma cumulativa
            soma_cumulativa = importancias_normalizadas.cumsum()

            print(soma_cumulativa)
            
            # 5. Encontrar os indices ate o primeiro que passa do limiar
            variaveis_selecionadas = soma_cumulativa[
                soma_cumulativa.shift(fill_value=0) < q
            ].index
            
            print(variaveis_selecionadas)
            
            matriz_estabilidade.loc[target, variaveis_selecionadas] += 1.0

        matriz_estabilidade.loc[target] = (
            matriz_estabilidade.loc[target] / iteracoes
        )
    fim = time.perf_counter()
    print(f'Matriz gerada em {fim - inicio:.4f} segundos!')
    return matriz_estabilidade

matriz = extrair_frequencia_estabilidade_grafo(df, df.columns, num_trees = 500, iteracoes = 100)

matriz.to_csv(
    f'reports/matriz_importancia_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
)