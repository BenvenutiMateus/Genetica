import pandas as pd
import os

df = pd.read_csv('data/bruto/banco_fenotipos_conformal.csv') 
df = df.drop(columns=['id', 'CP1','CP2','CP3','CP4','CP5','CP6'])
colunas_categoricas = ['fumo', 'raca_cor', 'conjugal3', 'trabalho2', 'bebem2']
df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int)
colunas_categoricas = [coluna for coluna in df.columns if df[coluna].nunique() < 6]
print(colunas_categoricas)
nrespondeu = [coluna for coluna in df.columns if 'respondeu' in coluna]
df = df.drop(columns=nrespondeu)
df = df.rename(columns={
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
    'fumo_fumante': 'fumo_fumante',
    'fumo_nunca': 'fumo_nunca',
    'raca_cor_branca': 'raca_cor_branca',
    'raca_cor_indígena': 'raca_cor_indigena',
    'raca_cor_outras': 'raca_cor_outras',
    'raca_cor_parda': 'raca_cor_parda',
    'raca_cor_preta': 'raca_cor_preta',
    'conjugal3_casado/un. estavel': 'estado_civil_casado_uniao_est',
    'conjugal3_separado/desq/divorc': 'estado_civil_separado_divorc',
    'conjugal3_solteiro': 'estado_civil_solteiro',
    'conjugal3_viuvo': 'estado_civil_viuvo',
    'trabalho2_estudante': 'trabalho_estudante',
    'trabalho2_nÃ£o trabalha': 'trabalho_nao_trabalha',
    'trabalho2_trabalha': 'trabalho_trabalha',
    'bebem2_NS/NR': 'consumo_alcool_ns_nr',
    'bebem2_atÃ© 3x semana': 'consumo_alcool_ate_3x_semana',
    'bebem2_nÃ£o bebe': 'consumo_alcool_nao_bebe'
})

df.to_csv('data/limpos/banco_fenotipos_conformal.csv', index = False)