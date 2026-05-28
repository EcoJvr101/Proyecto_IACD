import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    # 1. Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'cleaned_data.csv')

    try:
        df = pd.read_csv(input_path)
        print(f"Dados limpos carregados com sucesso. Dimensao inicial: {df.shape}")
    except FileNotFoundError:
        print(f"Erro: nao foi encontrado {input_path}.")
        return

    # 2. Limpieza previa del genero
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})

    # 3. Creacion de la variable objetivo de clasificacion
    # Se considera exito si el paciente pierde mas del 28% del peso inicial.
    # Este umbral genera una division equilibrada de las clases (cerca de 55/45).
    df['diet_success'] = (
        (-df['weight_change_kg_6m'] / df['baseline_weight_kg']) > 0.28
    ).astype(int)
    print(f"Distribuicao da classe sucesso: {df['diet_success'].mean()*100:.1f}% positivos")

    y = df['diet_success']

    # 4. Seleccion de variables
    # 4.1 Eliminar identificadores y columnas derivadas
    identificadores = ['patient_id', 'diet_id', 'nutritionist_id', 'program_id',
                       'record_created_at', 'cluster']
    cols_to_drop = [col for col in identificadores if col in df.columns]

    # 4.2 Eliminar variables que generan Data Leakage
    leakage_vars = ['adherence_ratio', 'mean_adherence_pct',
                    'motivation_score_program', 'program_index']
    cols_to_drop.extend([col for col in leakage_vars if col in df.columns])

    # 4.3 Eliminar el peso perdido y la variable objetivo para no enviesar el modelo
    cols_to_drop.extend(['weight_change_kg_6m', 'diet_success'])

    X = df.drop(columns=cols_to_drop)

    # 5. Preprocesamiento (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    print(f"Dimensao de X apos One-Hot Encoding: {X.shape}")

    # 6. Division de los datos (50% train / 30% validacion / 10% test / 10% sin usar)
    X_resto, X_unused, y_resto, y_unused = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resto, y_resto, test_size=4/9, random_state=42, stratify=y_resto)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/4, random_state=42, stratify=y_temp)

    print(f"Tamanho treino: {len(X_train)}, validacao: {len(X_val)}, "
          f"teste: {len(X_test)}, sem usar: {len(X_unused)}")

    # 7. Escalado de caracteristicas (necesario para la Regresion Logistica)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # 8. Entrenamiento de los modelos
    # Se da preferencia al arbol de decision por ser interpretable, como pide el enunciado.
    arvore = DecisionTreeClassifier(max_depth=5, random_state=42)
    logistica = LogisticRegression(max_iter=1000)
    floresta = RandomForestClassifier(n_estimators=200, random_state=42)

    arvore.fit(X_train, y_train)
    logistica.fit(X_train_sc, y_train)
    floresta.fit(X_train, y_train)

    # 9. Evaluacion en el conjunto de validacion
    modelos = {
        'Arvore de Decisao': (arvore, X_val, X_test),
        'Regressao Logistica': (logistica, X_val_sc, X_test_sc),
        'Random Forest': (floresta, X_val, X_test),
    }

    resultados = []
    for nome, (modelo, Xv, _) in modelos.items():
        y_pred = modelo.predict(Xv)
        resultados.append({
            'Modelo': nome,
            'Accuracy': accuracy_score(y_val, y_pred),
            'F1 Score': f1_score(y_val, y_pred),
        })

    resultados_df = pd.DataFrame(resultados)
    print("\n--- Tabela Comparativa de Classificacao (Validacao) ---")
    print(resultados_df.round(4).to_string(index=False))

    # 10. Evaluacion del mejor modelo en el conjunto de test
    melhor_nome = resultados_df.loc[resultados_df['F1 Score'].idxmax(), 'Modelo']
    melhor_modelo, _, X_test_eval = modelos[melhor_nome]
    y_pred_test = melhor_modelo.predict(X_test_eval)
    print(f"\nMelhor modelo: {melhor_nome}")
    print(f"Accuracy no teste: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1 Score no teste: {f1_score(y_test, y_pred_test):.4f}")
    print("Matriz de confusao (teste):")
    print(confusion_matrix(y_test, y_pred_test))

if __name__ == "__main__":
    main()
