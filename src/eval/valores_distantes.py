import pandas as pd
import numpy as np
import datetime

path = 'models/Importancias_2026-09-05_22-04-33/matriz_estabilidade_metodo_original.csv'
df = pd.read_csv(path, index_col = 0)
df_np = df.to_numpy()
dif = df_np - df_np.T
df = pd.DataFrame(
        dif,
        index=df.index,
        columns=df.columns
    )
df.to_csv(f'reports/diferenca_valores{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')