import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    # 1. Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'cleaned_data.csv')
    
    vis_dir = os.path.join(base_dir, 'visualizaciones')
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
        print(f"Datos limpios cargados exitosamente. Dimensión: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró {input_path}.")
        return

    # 2. Corrección de la columna 'sex'
    if 'sex' in df.columns:
        # Quitamos espacios, pasamos a mayúsculas
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        # Reemplazamos FEMALE por F y MALE por M
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})
        
        # Opcional: si existiera algún valor raro que no sea F o M, podríamos filtrarlo.
        # Aquí verificamos qué valores quedaron:
        print("\nValores únicos en 'sex' tras limpieza:", df['sex'].unique())

    # 3. Preparación para Clustering
    # Nota: verificamos si la columna se llama mean_adherence_pct o adherence_ratio
    adherence_col = 'mean_adherence_pct' if 'mean_adherence_pct' in df.columns else 'adherence_ratio'
    
    features = ['age', 'baseline_weight_kg', 'motivation_score', adherence_col, 'weight_change_kg_6m']
    
    # Validar que existan en el DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Faltan las siguientes columnas para hacer cluster: {missing_features}")
        return
        
    df_features = df[features].copy()
    
    # 4. Estandarización
    # Los algoritmos basados en distancias (como K-Means) son sensibles a la escala
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    
    # 5. Algoritmo K-Means
    # Utilizamos n_clusters=4 con una semilla (random_state) para reproducibilidad
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # 6. Análisis de los Perfiles
    cluster_summary = df.groupby('cluster')[features].mean()
    print("\n--- Perfiles de los Pacientes (Media por Cluster) ---")
    print(cluster_summary.round(2))
    
    print("\n--- Cantidad de pacientes por Cluster ---")
    print(df['cluster'].value_counts().sort_index())
    
    # 7. Visualización
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df, 
        x='baseline_weight_kg', 
        y='weight_change_kg_6m', 
        hue='cluster', 
        palette='Set1', 
        alpha=0.7,
        s=70
    )
    plt.title('Clusters de Pacientes: Peso Inicial vs Cambio de Peso', fontsize=14)
    plt.xlabel('Peso Inicial (kg)')
    plt.ylabel('Cambio de Peso a los 6 meses (kg)')
    # Asegurarnos de que la leyenda se lea como categórica
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_plot = os.path.join(vis_dir, 'clusters_peso.png')
    plt.savefig(out_plot)
    plt.close()
    print(f"\nVisualización de clusters guardada exitosamente en: {out_plot}")
    
    # Exportar el resultado con la nueva columna 'cluster'
    output_path = os.path.join(data_dir, 'clustered_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Datos etiquetados exportados en: {output_path}")

if __name__ == "__main__":
    main()
