import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    # 1. Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'cleaned_data.csv')

    try:
        df = pd.read_csv(input_path)
        print(f"Datos limpios cargados exitosamente. Dimension inicial: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontro {input_path}.")
        return

    # 2. Limpieza previa del genero
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})

    # 3. Creacion de la variable objetivo de clasificacion
    # Se considera exito si el paciente pierde mas del 28% de su peso inicial.
    # Este umbral genera una division equilibrada de las clases (cerca de 55/45).
    df['diet_success'] = (
        (-df['weight_change_kg_6m'] / df['baseline_weight_kg']) > 0.28
    ).astype(int)
    print(f"Distribucion de la clase exito: {df['diet_success'].mean()*100:.1f}% positivos")

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

    # 4.3 Eliminar el peso perdido y la variable objetivo para no hacer trampa
    cols_to_drop.extend(['weight_change_kg_6m', 'diet_success'])

    X = df.drop(columns=cols_to_drop)

    # 5. Preprocesamiento (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    print(f"Dimension de X despues de One-Hot Encoding: {X.shape}")

    # 6. Division de datos (50% train / 30% validacion / 10% test / 10% sin usar)
    X_resto, X_unused, y_resto, y_unused = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resto, y_resto, test_size=4/9, random_state=42, stratify=y_resto)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/4, random_state=42, stratify=y_temp)

    print(f"Tamano train: {len(X_train)}, validacion: {len(X_val)}, "
          f"test: {len(X_test)}, sin usar: {len(X_unused)}")

    # 7. Entrenamiento de modelos
    # Se da preferencia al arbol de decision por ser interpretable, como pide el enunciado.
    models = {
        'Arbol de Decision': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Regresion Logistica': LogisticRegression(max_iter=5000),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results.append({
            'Modelo': name,
            'Accuracy': accuracy_score(y_val, y_pred),
            'F1 Score': f1_score(y_val, y_pred)
        })

    results_df = pd.DataFrame(results)
    print("\n--- Tabla Comparativa de Clasificacion (Validacion) ---")
    print(results_df.round(4).to_string(index=False))

    # 8. Evaluacion del mejor modelo en el conjunto de test
    mejor_nombre = results_df.loc[results_df['F1 Score'].idxmax(), 'Modelo']
    mejor_modelo = models[mejor_nombre]
    y_pred_test = mejor_modelo.predict(X_test)
    print(f"\nMejor modelo: {mejor_nombre}")
    print(f"Accuracy en test: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1 Score en test: {f1_score(y_test, y_pred_test):.4f}")
    print("Matriz de confusion (test):")
    print(confusion_matrix(y_test, y_pred_test))

if __name__ == "__main__":
    main()
