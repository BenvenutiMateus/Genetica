from src.utils.transformacoes import imp_to_adj
import pandas as pd
import datetime
import numpy as np

df = pd.read_csv('reports/matriz_importancia_2026-08-29_16-38-09.csv', index_col = 0)

df = imp_to_adj(df)

df_novo = (df > 0.8).astype(int)

df = pd.DataFrame(df_transformado, columns=df.columns, index=df.index)
df.to_csv(f'reports/adj_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')