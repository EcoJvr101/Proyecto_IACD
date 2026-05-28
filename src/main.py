import os
import sys
import data_cleaning
import data_integration
import eda_analysis
import clustering
import predictive_modeling
import classification_model

def main():
    print("="*40)
    print(f"PIPELINE DE CIÊNCIA DE DADOS")
    print("="*40)

    try:
        
        #Data Integration
        print("\n>>> Integração de Dados <<<\n")
        data_integration.main()

        #Data Cleaning
        print("\n>>> Limpeza de Dados <<<\n")
        data_cleaning.main()

        #Data Analysis
        print("\n>>> Análise EDA <<<\n")
        eda_analysis.main()

        #Clasification Model - Arboles de decision
        print("\n>>> Clasification Model <<<")
        classification_model.main()

        #Clustering
        print("\n>>> Algoritmo K-Means <<<\n")
        clustering.main()

        #PRedictive Model
        print("\n>>> Modelo Preditivo <<<\n")
        predictive_modeling.main()

        print("="*90)
        print(f"PIPELINE DE CIÊNCIA DE DADOS EXECUTADO SEM ERROS")
        print(f"Representação visual adicional dos dados localizada em 'visualizaciones'")
        print("="*90)
    
    except Exception as e:
        print(f"\n[ERRO]: pipeline interrompido em {e}]\n")
        sys.exit(1)

if __name__ == "__main__":
    main()