import os
import pandas as pd

# 1. Leitura dos dados
df = pd.read_csv('data/bruto/banco_fenotipos_conformal.csv')

# 2. Remoção de colunas desnecessárias
df = df.drop(columns=['id', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'])

# 3. Codificação numérica das variáveis categóricas (mantém 1 coluna por variável)
colunas_categoricas = ['fumo', 'raca_cor', 'conjugal3', 'trabalho2', 'bebem2']

for col in colunas_categoricas:
  if col in df.columns:
    df[col] = df[col].astype('category').cat.codes

# 4. Remoção de colunas com "respondeu"
nrespondeu = [coluna for coluna in df.columns if 'respondeu' in coluna]
df = df.drop(columns=nrespondeu)

# 5. Renomeação das colunas (ajustado para as colunas originais)
df = df.rename(
    columns={
        'hb_g_dl': 'hemoglobina_g_dl',
        'glicose_mg_dl': 'glicose_mg_dl',
        'sexo': 'sexo',
        'idade': 'idade',
        'faixa_etaria': 'faixa_etaria',
        'imc': 'imc',
        'imc_cat4_aferido': 'imc_categoria_4',
        'imc_cat_aferido': 'imc_categoria',
        'ep': 'excesso_peso',
        'obesidade': 'obesidade',
        'ccm': 'circunferencia_cintura_cm',
        'obes_central': 'obesidade_central',
        'cc_estat': 'circ_cintura_estatura',
        'cc_estat_cat': 'circ_cintura_estatura_cat',
        'medDM_bioq': 'medicamento_dm_bioq',
        'diabetes2': 'diabetes_tipo_2',
        'HOMA_IR': 'homa_ir',
        'HOMA_cat': 'homa_categoria',
        'pas_final': 'pressao_sistolica_final',
        'pad_final': 'pressao_diastolica_final',
        'medHAS_bioq': 'medicamento_has_bioq',
        'has2': 'hipertensao_tipo_2',
        'col_nao_hdlc_mg_dl': 'colesterol_nao_hdl_mg_dl',
        'colnaohdl_cat': 'colesterol_nao_hdl_cat',
        # Nomes das colunas originais mantidas
        'fumo': 'fumo',
        'raca_cor': 'raca_cor',
        'conjugal3': 'estado_civil',
        'trabalho2': 'trabalho',
        'bebem2': 'consumo_alcool',
    }
)

# 6. Salvar o arquivo limpo
os.makedirs('data/limpos', exist_ok=True)
df.to_csv('data/limpos/banco_fenotipos_conformal.csv', index=False)