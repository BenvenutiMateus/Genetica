import pandas as pd
import os

def read_gene(caminho, genes = [i for i in range(1,23)]):
    COLUNAS_REMOCAO = ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    df_completo = None
    for arquivo in os.listdir(caminho):
        if arquivo.endswith('.raw'):
            gene = int(arquivo.split('_')[0].replace('chr',''))
            if gene in genes:
                df_gen = pd.read_csv(f'{caminho}/{arquivo}',sep=r'\s+')
                df_gen = df_gen.drop(columns=COLUNAS_REMOCAO)
                if df_completo is None:
                    df_completo = df_gen
                else:
                    df_completo = pd.merge(left=df_completo,right=df_gen,on='IID',how='inner')
    return df_completo

