import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset y realiza la limpieza final (eliminación de nulos en target e imputación).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe. Por favor, asegúrate de haber generado processed_data.csv primero.")
    
    df = pd.read_csv(filepath)
    
    # 1. Eliminar filas donde la variable objetivo es nula
    target_col = 'weight_change_kg_6m'
    if target_col in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=[target_col])
        print(f"Filas eliminadas por nulos en target: {initial_len - len(df)}")
    else:
        print(f"Advertencia: La columna objetivo '{target_col}' no se encuentra en el dataframe.")
        
    # 2. Imputar valores nulos en 'age' y 'sleep_hours' usando KNNImputer
    cols_to_impute = ['age', 'sleep_hours']
    # Solo imputamos si existen las columnas
    cols_present = [c for c in cols_to_impute if c in df.columns]
    
    if cols_present:
        # Usamos KNNImputer que suele ser más preciso que la mediana al usar otras variables para predecir
        imputer = KNNImputer(n_neighbors=5)
        # Es necesario aislar las columnas para no imputar todo el dataset si solo queremos estas
        df[cols_present] = imputer.fit_transform(df[cols_present])
        
    return df

def preprocess_for_clustering(df: pd.DataFrame, features: list) -> np.ndarray:
    """
    Selecciona las características relevantes y aplica normalización (StandardScaler).
    """
    # Filtrar solo las características que están en el dataframe para evitar KeyError
    valid_features = [f for f in features if f in df.columns]
    
    # Extraemos el subconjunto de datos
    X = df[valid_features].copy()
    
    # Manejar posibles valores nulos que hayan quedado en otras características por seguridad
    X = X.fillna(X.median(numeric_only=True)) 
    
    # 3. Escalar los datos para que el modelo no se sesgue hacia variables con magnitudes más altas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, valid_features

def find_optimal_k(X_scaled: np.ndarray, max_k: int = 10):
    """
    Determina el número óptimo de clusters usando el método del codo y Silhouette score.
    Nota: Esta función bloquea la ejecución para mostrar los gráficos.
    """
    inertia = []
    silhouette_scores = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
    # Visualización del Método del Codo
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, marker='o', linestyle='--')
    plt.title('Método del Codo (Elbow Method)')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.xticks(K_range)
    
    # Visualización del Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='--', color='green')
    plt.title('Coeficiente de Silhouette')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(K_range)
    
    plt.tight_layout()
    plt.show()

def apply_kmeans(df: pd.DataFrame, X_scaled: np.ndarray, optimal_k: int) -> pd.DataFrame:
    """
    Entrena el modelo K-Means con el k óptimo y añade la etiqueta al DataFrame.
    """
    # Entrenamiento definitivo con el k seleccionado
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    
    # 4. Añadir nueva columna al dataframe original
    df['cluster'] = kmeans.fit_predict(X_scaled)
    # Convertimos a string / category para que sea tratada como cualitativa en los gráficos
    df['cluster'] = 'Cluster ' + df['cluster'].astype(str)
    
    return df

def analyze_clusters(df: pd.DataFrame):
    """
    Genera visualizaciones para interpretar los clusters resultantes.
    """
    # Configuración de estilo de Seaborn
    sns.set_theme(style="whitegrid")
    
    # 1. Boxplot: Relación entre cluster y cambio de peso
    if 'weight_change_kg_6m' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='cluster', y='weight_change_kg_6m', palette='viridis', order=sorted(df['cluster'].unique()))
        plt.title('Distribución de Cambio de Peso por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Cambio de Peso a 6 meses (kg)')
        plt.show()
    
    # 2. Scatterplot: Edad vs Cambio de peso según el cluster
    if 'age' in df.columns and 'weight_change_kg_6m' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='age', y='weight_change_kg_6m', hue='cluster', palette='viridis', alpha=0.7, hue_order=sorted(df['cluster'].unique()))
        plt.title('Edad vs Cambio de Peso (Agrupado por Cluster)')
        plt.xlabel('Edad')
        plt.ylabel('Cambio de Peso a 6 meses (kg)')
        plt.legend(title='Grupo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # 3. Pairplot para ver cómo interactúan características clave
    # 'motivation_score' puede ser importante si existe en lugar de 'motivitation_score'
    features_for_pairplot = ['age', 'baseline_bmi', 'weight_change_kg_6m', 'cluster']
    valid_features = [f for f in features_for_pairplot if f in df.columns]
    
    if len(valid_features) > 2:
        sns.pairplot(df[valid_features], hue='cluster', palette='viridis', corner=True, plot_kws={'alpha': 0.6})
        plt.suptitle('Relaciones entre variables por Cluster', y=1.02)
        plt.show()

def main():
    # Ruta dinámica al archivo de datos processed_data.csv (se asume que se guarda en data/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'processed_data.csv')
    
    # Nota de pre-ejecución
    print(f"Buscando el dataset en: {os.path.normpath(data_path)}")
    
    # Características a utilizar en el clustering
    # Ajusta los nombres si difieren en el dataset real (ej: motivitation_score)
    clustering_features = [
        'age', 
        'baseline_bmi', 
        'carb_pct', 
        'motivation_score', 
        'weight_change_kg_6m'
    ]
    
    try:
        print("\n--- 1. Limpieza Final ---")
        df = load_and_clean_data(data_path)
        print(f"Dimensiones finales del dataset: {df.shape}")
        
        print("\n--- 2. Preprocesamiento para Clustering ---")
        X_scaled, used_features = preprocess_for_clustering(df, clustering_features)
        print(f"Características analizadas: {used_features}")
        
        # --- 3. Encontrar k óptimo ---
        # ATENCIÓN: Descomentar la siguiente línea para visualizar el codo y silhouette
        # Esta gráfica te ayudará a decidir el número 'optimal_k' a usar
        # print("\n--- Evaluando número óptimo de clusters (k) ---")
        # find_optimal_k(X_scaled, max_k=10)
        
        # Asumiendo que determinamos k=4 tras ver las métricas (ajustar este valor)
        optimal_k = 4  
        print(f"\n--- 3. Aplicando K-Means con k={optimal_k} ---")
        df = apply_kmeans(df, X_scaled, optimal_k)
        
        print("\n--- 4. Análisis y Visualización de Grupos ---")
        analyze_clusters(df)
        
        # Guardado opcional de los datos con clusters asignados
        output_path = os.path.join(base_dir, '..', 'data', 'clustered_data.csv')
        df.to_csv(output_path, index=False)
        print(f"\n[Éxito] Proceso finalizado. Dataset con etiquetas de clusters guardado en: {os.path.normpath(output_path)}")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Asegúrate de ejecutar tu script principal para crear y guardar 'processed_data.csv' (ej: dataframe.to_csv('data/processed_data.csv', index=False))")
    except Exception as e:
        print(f"\n[Error inesperado] {e}")

if __name__ == "__main__":
    main()
