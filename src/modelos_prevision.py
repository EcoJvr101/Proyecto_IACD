import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset y realiza la limpieza y codificación de variables categóricas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 1. Eliminar columnas que no sirven para predecir (IDs y fechas)
    cols_to_drop = [
        'patient_id', 'diet_id', 'nutritionist_id', 
        'program_id', 'record_created_at', 'diet_name'
    ]
    # Eliminamos solo las que existan en el DataFrame para evitar errores
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)
    
    # 2. Identificar y codificar variables categóricas (One-Hot Encoding)
    categorical_cols = ['sex', 'diet_type', 'approach', 'specialty', 'cluster']
    categorical_cols_existing = [c for c in categorical_cols if c in df.columns]
    
    # pd.get_dummies convierte categóricas en múltiples columnas binarias (0 y 1)
    df_encoded = pd.get_dummies(df, columns=categorical_cols_existing, drop_first=True)
    
    # Rellenar posibles nulos restantes con la mediana por seguridad
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    
    return df_encoded

def evaluate_model(y_true, y_pred, model_name: str):
    """
    Calcula e imprime MAE, RMSE y R2 para las predicciones dadas.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- Resultados para {model_name} ---")
    print(f"MAE (Error Absoluto Medio): {mae:.4f} kg")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f} kg")
    print(f"R² Score: {r2:.4f}\n")
    
    return mae, rmse, r2

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Extrae la importancia de las variables del modelo de bosque aleatorio 
    y grafica el Top N.
    """
    # Extraer las importancias
    importances = model.feature_importances_
    
    # Crear un DataFrame para facilitar el ordenamiento y graficado
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Seleccionar solo el Top N
    top_features = feat_imp_df.head(top_n)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=top_features, 
        x='Importance', 
        y='Feature', 
        palette='viridis'
    )
    plt.title(f'Top {top_n} Factores que influyen en el Éxito de la Dieta')
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()

def main():
    # Rutas relativas robustas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'clustered_data.csv')
    
    try:
        print("1. Cargando y preparando datos...")
        df = load_and_prepare_data(data_path)
        print(f"Datos listos. Dimensiones tras One-Hot Encoding: {df.shape}")
        
        # 3. Separar Variable Objetivo (y) y Características (X)
        target_col = 'weight_change_kg_6m'
        if target_col not in df.columns:
            raise ValueError(f"No se encontró la variable objetivo '{target_col}'")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 4. División Train / Test (80% / 20%)
        print("\n2. Dividiendo datos en Train (80%) y Test (20%)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Set de Entrenamiento: {X_train.shape[0]} filas")
        print(f"Set de Prueba: {X_test.shape[0]} filas")
        
        # 5. Entrenar y Evaluar Ridge Regression (Modelo Lineal Regularizado)
        print("\n3. Entrenando modelos...\n")
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)
        evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
        
        # 6. Entrenar y Evaluar RandomForestRegressor (Ensamble)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")
        
        # 7. Graficar Importancia de Características (Feature Importance)
        print("4. Generando gráfico de Feature Importance...")
        plot_feature_importance(rf_model, X_train.columns, top_n=10)
        
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Asegúrate de ejecutar primero el script de clustering para generar 'clustered_data.csv'.")
    except Exception as e:
        print(f"\n[Error inesperado] {e}")

if __name__ == "__main__":
    main()
