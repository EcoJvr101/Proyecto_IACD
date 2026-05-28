Elaborado por: 

Leandro Apolo N: 55459
Jorge Altamirano N: 55456
Jose Gordon N: 55432


# Projeto IACD

Este repositório contém um pipeline completo de Ciência de Dados desenvolvido em **Python** para integração, limpeza, análise exploratória, agrupamento (clustering) e modelagem preditiva baseados em dados de pacientes submetidos a programas de perda de peso orientados por nutricionistas.

O principal objetivo deste projeto é compreender os fatores que influenciam a perda de peso dos pacientes ao fim de 6 meses (`weight_change_kg_6m`) e construir um modelo preditivo capaz de estimar este resultado com base no perfil inicial dos pacientes e nas dietas prescritas.

---

##  Estrutura do Projeto

A organização dos arquivos no repositório segue a estrutura abaixo:

```text
Proyecto_IACD/
├── data/                       # Arquivos de dados (CSVs)
│   ├── diets.csv               # Informações sobre os tipos de dietas
│   ├── nutritionists.csv       # Cadastro e especialidades dos nutricionistas
│   ├── patients.csv            # Dados demográficos e físicos dos pacientes
│   ├── outcomes.csv            # Resultados clínicos e de adesão dos programas
│   ├── merged_data.csv         # [Gerado] Dados unificados após a integração
│   ├── cleaned_data.csv        # [Gerado] Dados limpos e tratados
│   └── clustered_data.csv      # [Gerado] Dados rotulados com os grupos/clusters
│
├── src/                        # Código-fonte do pipeline
│   ├── main.py                 # Orquestrador do pipeline completo
│   ├── data_integration.py     # Fase 1: Integração das fontes de dados
│   ├── data_cleaning.py        # Fase 2: Limpeza, imputação e tratamento de outliers
│   ├── eda_analysis.py         # Fase 3: Análise Exploratória de Dados (EDA)
│   ├── clustering.py           # Fase 4: Agrupamento de pacientes com K-Means
│   └── predictive_modeling.py  # Fase 5: Algoritmos de Regressão e Modelagem
│
├── visualizaciones/            # [Gerado] Gráficos estatísticos gerados no pipeline
│   ├── heatmap_correlacion.png # Mapa de calor da correlação de Pearson
│   ├── boxplot_dietas.png      # Distribuição de perda de peso por tipo de dieta
│   ├── barras_nutricionista.png# Eficácia média por abordagem do nutricionista
│   ├── boxplot_sexo.png        # Comparação de perda de peso entre sexos
│   └── clusters_peso.png       # Dispersão do peso inicial vs. perda de peso por cluster
│
└── Relatorio_IACD/             # Documentação técnica e relatórios finais
    └── Relatorio_IACD.zip      # Arquivo comprimido com o relatório do projeto
```

---

## Fases do Pipeline de Dados

O projeto está estruturado em módulos independentes que são orquestrados sequencialmente:

### 1. Integração de Dados (`data_integration.py`)
Esta fase carrega as 4 fontes de dados separadas em arquivos CSV e realiza a consolidação em um único conjunto de dados centralizado.
* **Processo**: Executa junções à esquerda (*Left Merges*) utilizando a base de resultados clínicos (`outcomes.csv`) como tabela âncora, cruzando as chaves `patient_id`, `diet_id` e `nutritionist_id`.
* **Saída**: Gera o arquivo `data/merged_data.csv`.

### 2. Limpeza de Dados e Pré-processamento (`data_cleaning.py`)
Tratamento e higienização do dataset integrado para garantir a qualidade dos modelos analíticos.
* **Padronização**: Padroniza os valores de gênero na coluna `sex` para `F` (Feminino) e `M` (Masculino) e remove espaços ou formatações inconsistentes em campos categóricos (`diet_name`, `diet_type`, `approach`, `specialty`).
* **Redundâncias**: Remove atributos duplicados ou desnecessários, como `bmi_redundant` e `experience_years`.
* **Valores Ausentes (Missing Values)**:
  * Remove linhas sem valores nas variáveis críticas de interesse: variação de peso (`weight_change_kg_6m`) e taxa de adesão (`adherence_ratio`).
  * Imputa valores ausentes de variáveis numéricas usando a **mediana**.
  * Imputa valores ausentes de variáveis categóricas usando a **moda**.
* **Outliers**: Deteta e suaviza valores discrepantes de idade (`age`), peso inicial (`baseline_weight_kg`) e altura (`height_cm`) utilizando o **método IQR** combinado com a técnica de **Capping** (limitação de valores extremos aos limites superior e inferior).
* **Saída**: Gera o arquivo `data/cleaned_data.csv`.

###  3. Análise Exploratória de Dados - EDA (`eda_analysis.py`)
Módulo dedicado a extrair percepções de negócio e entender a distribuição das variáveis.
* **Correlação**: Calcula a matriz de correlação de Pearson entre as variáveis numéricas e exibe no console as 5 variáveis com maior correlação linear com a variação de peso final.
* **Visualizações**: Cria e salva gráficos ricos em `visualizaciones/` para análise de hipóteses:
  * *Mapa de Calor*: Correlação visual de todas as métricas numéricas.
  * *Boxplot de Dietas*: Variação de peso por tipo de dieta (Keto, Low-Carb, Vegana, etc.).
  * *Gráfico de Barras*: Desempenho médio baseado na abordagem de acompanhamento do nutricionista (Comportamental, Restritiva, Flexível, etc.).
  * *Boxplot por Gênero*: Distribuição de perda de peso entre homens e mulheres.
* **Insights**: Fornece respostas automáticas no console sobre a dieta com maior perda média, a abordagem mais eficaz do profissional de nutrição e diferenças significativas entre sexos.

### 👥 4. Agrupamento - Clustering (`clustering.py`)
Identificação de perfis e segmentos de pacientes a partir de comportamento homogêneo.
* **Algoritmo**: Utiliza o **K-Means** configurado para segmentar a base em **4 clusters** de pacientes.
* **Atributos Utilizados**: Idade (`age`), peso inicial (`baseline_weight_kg`), pontuação de motivação (`motivation_score`), nível de adesão (`adherence_ratio` ou `mean_adherence_pct`) e alteração de peso final (`weight_change_kg_6m`).
* **Preparações**: Aplica normalização por **StandardScaler** para padronizar escalas, já que o algoritmo é altamente sensível à distância.
* **Saída**: Produz estatísticas de médias por cluster para traçar perfis demográficos/de aderência e gera o arquivo rotulado `data/clustered_data.csv` junto ao gráfico `visualizaciones/clusters_peso.png`.

### 5. Modelagem Preditiva (`predictive_modeling.py`)
Construção de modelos matemáticos para estimar a perda de peso a longo prazo.
* **Mitigação de Data Leakage (Vazamento de Dados)**: Remove variáveis que revelam dados coletados ao longo ou após o programa (como taxa de adesão ou índices específicos de acompanhamento), garantindo que o modelo preveja apenas com as informações disponíveis no **momento inicial** da consulta.
* **Engenharia de Recursos (One-Hot Encoding)**: Transforma variáveis categóricas em representações numéricas binárias (dummies) apropriadas para regressão.
* **Divisão de Dados**: Separa o conjunto em **80% para treino** e **20% para teste**, permitindo uma validação fidedigna.
* **Modelos Treinados e Avaliados**:
  1. *Regressão Linear Múltipla* (Modelo de base estatística clássico)
  2. *Random Forest Regressor* (Conjunto de árvores de decisão)
  3. *Gradient Boosting Regressor* (Modelagem iterativa de otimização de erros)
* **Avaliação**: Gera uma tabela comparativa com as métricas **MSE** (Erro Quadrático Médio), **RMSE** (Raiz do Erro Quadrático Médio) e **R² Score** (Coeficiente de Determinação) para facilitar a escolha do melhor algoritmo preditivo.

---

##  Como Executar o Pipeline

###  Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado e as bibliotecas científicas necessárias. Você pode instalá-las executando:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Executando o Pipeline Completo
Para rodar toda a esteira de dados de forma automatizada do início ao fim (Integração  Limpeza  EDA  Clustering  Modelagem):

```bash
python src/main.py
```

Durante a execução, o terminal exibirá logs detalhados de cada etapa, dimensões dos dados gerados, principais correlações estatísticas, análises comparativas e a tabela de acurácia dos modelos preditivos.

---

##  Tecnologias Utilizadas

* **Python** (Linguagem Principal)
* **Pandas** e **NumPy** (Manipulação e Engenharia de Dados)
* **Matplotlib** e **Seaborn** (Visualização Estática de Dados)
* **Scikit-Learn** (Pré-processamento, K-Means e Modelos de Regressão Machine Learning)
