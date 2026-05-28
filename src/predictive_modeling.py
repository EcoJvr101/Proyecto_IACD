import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'cleaned_data.csv')
    
    try:
        df = pd.read_csv(input_path)
        print(f"Datos limpios cargados exitosamente. Dimensión inicial: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró {input_path}.")
        return

    # 2. Limpieza previa del Género
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})

    # 3. Selección de Variables
    target = 'weight_change_kg_6m'
    if target not in df.columns:
        print(f"Error: La variable objetivo '{target}' no se encuentra en el dataset.")
        return
        
    y = df[target]
    
    # 3.1 Eliminar variables identificadoras y la columna cluster
    identificadores = ['patient_id', 'diet_id', 'nutritionist_id', 'program_id', 'record_created_at', 'cluster']
    cols_to_drop = [col for col in identificadores if col in df.columns]
    
    # 3.2 Eliminar variables que generan Data Leakage
    leakage_vars = ['adherence_ratio', 'mean_adherence_pct', 'motivation_score_program', 'program_index']
    cols_to_drop.extend([col for col in leakage_vars if col in df.columns])
    
    # Adicionamos el target a la lista de columnas a eliminar para aislar a X
    cols_to_drop.append(target)
    
    # Crear características predictoras (X)
    X = df.drop(columns=cols_to_drop)
    
    # 4. Preprocesamiento (One-Hot Encoding)
    # pd.get_dummies detectará automáticamente las variables object/category y las codificará
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Dimensión de X (características) después de One-Hot Encoding: {X.shape}")

    # 5. División de datos (50% train / 30% validación / 10% test / 10% sin usar)
    # Esquema indicado en el enunciado. El 10% final se reserva sin usar para evitar overfitting.
    # Primero separamos el 10% no utilizado
    X_resto, X_unused, y_resto, y_unused = train_test_split(
        X, y, test_size=0.10, random_state=42)
    # Del 90% restante: 50/30/10 equivale a 5/9, 3/9, 1/9
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resto, y_resto, test_size=4/9, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/4, random_state=42)

    print(f"Tamano train: {len(X_train)}, validacion: {len(X_val)}, "
          f"test: {len(X_test)}, sin usar: {len(X_unused)}")

    # 6. Entrenamiento de Modelos
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }
    
    results = []

    # 7. Evaluación sobre el conjunto de validación
    for name, model in models.items():
        # Entrenar el modelo con el conjunto de entrenamiento
        model.fit(X_train, y_train)
        
        # Predecir sobre el conjunto de validación
        y_pred = model.predict(X_val)
        
        # Calcular métricas
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'R2 Score': r2
        })
        
    # Crear y mostrar tabla comparativa
    results_df = pd.DataFrame(results)
    
    print("\n--- Tabla Comparativa de Rendimiento de Modelos (Validacion) ---")
    print(results_df.round(4).to_string(index=False))

    # Evaluación final del mejor modelo sobre el conjunto de test
    mejor_nombre = results_df.loc[results_df['R2 Score'].idxmax(), 'Model']
    mejor_modelo = models[mejor_nombre]
    y_pred_test = mejor_modelo.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"\nMejor modelo: {mejor_nombre}")
    print(f"Resultado en test -> R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

if __name__ == "__main__":
    main()
