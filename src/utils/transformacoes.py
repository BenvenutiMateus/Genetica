import numpy as np
import pandas as pd


def imp_to_adj(df):
    matriz = df.to_numpy()

    matriz_min = np.minimum(matriz, matriz.T)

    return pd.DataFrame(
        matriz_min,
        index=df.index,
        columns=df.columns
    )