import pandas as pd
import os
import json
import time
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import time
from src.utils.importar_genotipo import read_gene
from tqdm import tqdm

def extrair_importancias_brutas(
    df, alvos, num_trees=500, iteracoes=50, fracao_amostra=0.5, pasta=None, n_permut = 5
):
    """
    Roda RF + permutation importance pra cada alvo/iteração e salva a
    importância BRUTA (sem normalizar) em um CSV por variável-alvo,
    formato longo. Salva também um metadata.json com os parâmetros da
    rodada. Retoma automaticamente: pula iterações já salvas.
    """
    if pasta is None:
        pasta = f'models/Importancias_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    pasta_importancias = os.path.join(pasta, 'importancias')
    os.makedirs(pasta_importancias, exist_ok=True)

    caminho_metadata = os.path.join(pasta, 'metadata.json')

    metadata = {
        'num_trees': num_trees,
        'iteracoes': iteracoes,
        'fracao_amostra': fracao_amostra,
        'alvos': list(alvos),
        'colunas_df': list(df.columns),
        'n_linhas_df': len(df),
        'inicio': datetime.datetime.now().isoformat(),
    }

    with open(caminho_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    inicio = time.perf_counter()

    colunas_categoricas = [
        c for c in df.columns
        if df[c].nunique() < 5
    ]

    # Barra geral
    barra_geral = tqdm(
        total=len(alvos) * iteracoes,
        desc="Progresso geral",
        unit="iteração"
    )

    for target in alvos:

        arquivo = os.path.join(
            pasta_importancias,
            f"{target}.csv"
        )

        df_alvo = df.dropna(subset=[target])

        # Descobrir iterações já salvas
        if os.path.exists(arquivo):
            iteracoes_existentes = set(
                pd.read_csv(
                    arquivo,
                    usecols=['iteracao']
                )['iteracao'].unique()
            )
        else:
            iteracoes_existentes = set()

        # Barra específica do alvo
        for i in tqdm(
            range(iteracoes),
            desc=f"{target}",
            unit="it",
            leave=False
        ):

            if i in iteracoes_existentes:
                barra_geral.update(1)
                continue

            if fracao_amostra > 1:
                df_sub = df_alvo.sample(
                    n=fracao_amostra,
                    random_state=i
                )
            else:
                df_sub = df_alvo.sample(
                    frac=fracao_amostra,
                    random_state=i
                )

            X = df_sub.drop(columns=[target])
            y = df_sub[target]

            if target not in colunas_categoricas:

                modelorf = RandomForestRegressor(
                    n_estimators=num_trees,
                    max_features=0.3,
                    n_jobs=-1,
                    random_state=i
                )

            else:

                modelorf = RandomForestClassifier(
                    n_estimators=num_trees,
                    max_features=0.3,
                    n_jobs=-1,
                    random_state=i
                )

            modelorf.fit(X, y)

            resultado_permutacao = permutation_importance(
                modelorf,
                X,
                y,
                n_repeats= n_permut,
                random_state=i,
                n_jobs=-1
            )

            importancias_brutas = pd.Series(
                resultado_permutacao.importances_mean,
                index=X.columns
            ).clip(lower=0)

            linha = pd.DataFrame({
                'iteracao': i,
                'variavel': importancias_brutas.index,
                'importancia_bruta': importancias_brutas.values,
                'importancia_std': resultado_permutacao.importances_std,
                'seed': i,
                'num_trees': num_trees,
                'fracao_amostra': fracao_amostra,
                'timestamp': datetime.datetime.now().isoformat(),
            })

            escrever_cabecalho = not os.path.exists(arquivo)

            linha.to_csv(
                arquivo,
                mode='a',
                header=escrever_cabecalho,
                index=False
            )

            barra_geral.update(1)

    barra_geral.close()

    metadata['fim'] = datetime.datetime.now().isoformat()
    metadata['numero_permutacao'] = n_permut
    with open(caminho_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    fim = time.perf_counter()

    print(
        f'Importâncias brutas geradas em '
        f'{fim - inicio:.4f}s! '
        f'Salvas em: {pasta}'
    )

    return pasta

def extrair_importancias_brutas_gini(
    df, alvos, num_trees=500, iteracoes=50, fracao_amostra=0.5, pasta=None, n_permut = 5
):
    """
    Roda RF + gini importance pra cada alvo/iteração e salva a
    importância BRUTA (sem normalizar) em um CSV por variável-alvo,
    formato longo. Salva também um metadata.json com os parâmetros da
    rodada. Retoma automaticamente: pula iterações já salvas.
    """
    if pasta is None:
        pasta = f'models/Importancias_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    pasta_importancias = os.path.join(pasta, 'importancias')
    os.makedirs(pasta_importancias, exist_ok=True)

    caminho_metadata = os.path.join(pasta, 'metadata.json')

    metadata = {
        'num_trees': num_trees,
        'iteracoes': iteracoes,
        'fracao_amostra': fracao_amostra,
        'alvos': list(alvos),
        'colunas_df': list(df.columns),
        'n_linhas_df': len(df),
        'inicio': datetime.datetime.now().isoformat(),
    }

    with open(caminho_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    inicio = time.perf_counter()

    colunas_categoricas = [
        c for c in df.columns
        if df[c].nunique() < 6
    ]

    # Barra geral
    barra_geral = tqdm(
        total=len(alvos) * iteracoes,
        desc="Progresso geral",
        unit="iteração"
    )

    for target in alvos:

        arquivo = os.path.join(
            pasta_importancias,
            f"{target}.csv"
        )

        df_alvo = df.dropna(subset=[target])

        # Descobrir iterações já salvas
        if os.path.exists(arquivo):
            iteracoes_existentes = set(
                pd.read_csv(
                    arquivo,
                    usecols=['iteracao']
                )['iteracao'].unique()
            )
        else:
            iteracoes_existentes = set()

        # Barra específica do alvo
        for i in tqdm(
            range(iteracoes),
            desc=f"{target}",
            unit="it",
            leave=False
        ):

            if i in iteracoes_existentes:
                barra_geral.update(1)
                continue

            if fracao_amostra > 1:
                df_sub = df_alvo.sample(
                    n=fracao_amostra,
                    random_state=i
                )
            else:
                df_sub = df_alvo.sample(
                    frac=fracao_amostra,
                    random_state=i
                )

            X = df_sub.drop(columns=[target])
            y = df_sub[target]

            if target not in colunas_categoricas:

                modelorf = RandomForestRegressor(
                    n_estimators=num_trees,
                    max_features=0.3,
                    n_jobs=-1,
                    bootstrap=False,
                    random_state=i
                )

            else:

                modelorf = RandomForestClassifier(
                    n_estimators=num_trees,
                    max_features=0.3,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=i
                )

            modelorf.fit(X, y)


            importancias_brutas = pd.Series(
                modelorf.feature_importances_,
                index=X.columns
            )

            linha = pd.DataFrame({
                'iteracao': i,
                'variavel': importancias_brutas.index,
                'importancia_bruta': importancias_brutas.values,
                'seed': i,
                'num_trees': num_trees,
                'fracao_amostra': fracao_amostra,
                'timestamp': datetime.datetime.now().isoformat()
            })

            escrever_cabecalho = not os.path.exists(arquivo)

            linha.to_csv(
                arquivo,
                mode='a',
                header=escrever_cabecalho,
                index=False
            )

            barra_geral.update(1)

    barra_geral.close()

    metadata['fim'] = datetime.datetime.now().isoformat()
    metadata['numero_permutacao'] = n_permut
    with open(caminho_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    fim = time.perf_counter()

    print(
        f'Importâncias brutas geradas em '
        f'{fim - inicio:.4f}s! '
        f'Salvas em: {pasta}'
    )

    return pasta

def calcular_matriz_estabilidade(pasta, alvos=None, todas_colunas=None, q=3, p=0.8, iteracoes=None):
    """
    Monta a matriz de estabilidade a partir dos CSVs brutos já salvos,
    sem rodar nenhum modelo de novo. Normaliza e aplica o corte `q` e `p`
    aqui, então é possível testar valores diferentes de `p` e `q` e
    `iteracoes` livremente.
    """
    caminho_metadata = os.path.join(pasta, 'metadata.json')
    if alvos is None or todas_colunas is None:
        with open(caminho_metadata, 'r') as f:
            metadata = json.load(f)

        if alvos is None:
            alvos = metadata['alvos']

        if todas_colunas is None:
            todas_colunas = metadata.get('colunas_df', alvos)

    pasta_importancias = os.path.join(pasta, 'importancias')
    matriz_estabilidade = pd.DataFrame(
        index=alvos, columns=todas_colunas, dtype=np.float64
    ).fillna(0.0)
    inicio = time.perf_counter()
    for target in alvos:
        arquivo = os.path.join(pasta_importancias, f"{target}.csv")
        if not os.path.exists(arquivo):
            print(f"Aviso: nenhum arquivo encontrado para {target}")
            continue

        dados = pd.read_csv(arquivo)

        iteracoes_disponiveis = sorted(dados['iteracao'].unique())
        if iteracoes is not None:
            iteracoes_disponiveis = iteracoes_disponiveis[:iteracoes]

        n_usadas = 0
        for i in iteracoes_disponiveis:
            grupo = dados[dados['iteracao'] == i].set_index('variavel')['importancia_bruta']
            grupo_ordenado = grupo.sort_values(ascending=False)

            soma_total = grupo_ordenado.sum()
            importancias_normalizadas = (
                grupo_ordenado / soma_total if soma_total > 0 else grupo_ordenado
            )

            soma_cumulativa = importancias_normalizadas.cumsum()
            variaveis_selecionadas = soma_cumulativa[
                soma_cumulativa.shift(fill_value=0) < p
            ].index

            if len(variaveis_selecionadas) < q:
                variaveis_selecionadas = soma_cumulativa.index[:q]

            matriz_estabilidade.loc[target, variaveis_selecionadas] += 1.0
            n_usadas += 1

        if n_usadas > 0:
            matriz_estabilidade.loc[target] = matriz_estabilidade.loc[target] / n_usadas
    fim = time.perf_counter()
    print(f'Matriz gerada em {fim - inicio:.4f} segundos!')
    return matriz_estabilidade

if __name__ == "__main__":
    df = pd.read_csv('data/limpos/banco_fenotipos_conformal.csv')
    
    df = df.drop(columns = ['samplefilename'])
    extrair_importancias_brutas(df, df.columns, iteracoes = 50)
    
    # caminho = 'sintetics'
    # matriz = calcular_matriz_estabilidade(caminho, q = 4, p = 0.1)
    
    # matriz.to_csv(os.path.join(caminho, 'matriz_estabilidade_metodo_original.csv'))