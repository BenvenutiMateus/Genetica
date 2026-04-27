Predição de Níveis de Glicose via Random Forest em Dados Genômicos
Este repositório contém o pipeline de análise estatística desenvolvido para a analise de marcadores genéticos (SNPs) e variáveis sociodemográficas. O projeto é parte de uma pesquisa de Iniciação Científica do departamento de Estatística da UFSCar.

Estrutura dos Dados
O projeto integra três fontes distintas de dados, harmonizadas através do identificador único do indivíduo.

1. Banco Genético Imputado LD
Localizado na pasta Banco Genetico Imputado LD, contém arquivos no formato .raw (output do PLINK) para cada cromossomo.

Nomenclatura: chr{num}_imputado_LD.raw

Amostra: 720 indivíduos.

Metadados: As primeiras seis colunas são identificadores de pedigree e fenótipo. A coluna IID é a chave primária utilizada para a ordenação e integração dos dados.

Preditores Genéticos: SNPs codificados numericamente (ex: rs62224618_T).

2. Mapas de Marcadores
Arquivos que descrevem a localização dos SNPs:

Nomenclatura: chr{num}map_imputado_LD.map

Conteúdo: Código do cromossomo, Nome do SNP, Posição em Morgans e Coordenada do par de bases.

3. Fenótipos e Covariáveis
Arquivo banco_fenotipos_conformal.csv contendo os desfechos e variáveis clínicas.

Variável Chave: samplefilename, que deve ser obrigatoriamente pareada e ordenada com a variável IID do banco genético.

Variáveis Categóricas: Inclui dados sobre fumo, raça/cor, estado conjugal, ocupação e consumo de álcool.

Metodologia Estatística
O processamento e análise seguem rigor estatístico para lidar com a alta dimensionalidade (p >> n):

Pré-processamento: Descrito detalhadamente na pasta info_pre_processamento, incluindo critérios de imputação e controle de qualidade.

Codificação: Variáveis categóricas tratadas via One-Hot Encoding para evitar hierarquias arbitrárias em dados nominais.

Modelo: Implementação do algoritmo Random Forest Regressor.

Configuração: 500 árvores de decisão.

Otimização: Uso de max_features ajustado para lidar com a dimensionalidade de aproximadamente 244.000 colunas.

Validação: Estimativa de erro via Out-of-Bag (OOB) Score e análise de importância de variáveis (Feature Importance) baseada na redução da variância residual.

Observações sobre os Arquivos
Devido às restrições de armazenamento do GitHub e à natureza sensível dos dados genômicos, os arquivos brutos (.raw e .csv) não estão hospedados neste repositório. O código está configurado para ler a estrutura de pastas local conforme descrito acima.

Contato
Para dúvidas sobre a metodologia ou acesso aos dados brutos para fins de reprodução acadêmica, favor entrar em contato através do e-mail disponível no perfil deste GitHub.
