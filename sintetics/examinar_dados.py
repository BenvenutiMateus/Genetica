from src.models.gerar_arquivos_importancia import extrair_importancias_brutas_gini
import pandas as pd
import numpy as np
df = pd.read_csv('sintetics/dados_sinteticos.csv')
extrair_importancias_brutas_gini(df, df.columns, pasta = 'sintetics_gini', iteracoes=50)