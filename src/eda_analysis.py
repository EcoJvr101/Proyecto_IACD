import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1. Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'cleaned_data.csv')
    
    # Crear carpeta para visualizaciones si no existe
    vis_dir = os.path.join(base_dir, 'visualizaciones')
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
        print(f"Datos limpios cargados exitosamente. Dimensión: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {input_path}.")
        return

    # 2. Matriz de Correlación y Top 5 Variables
    # Seleccionar solo variables numéricas
    num_df = df.select_dtypes(include=[np.number])
    
    # Calcular matriz de correlación de Pearson
    corr_matrix = num_df.corr()
    
    target = 'weight_change_kg_6m'
    if target in corr_matrix.columns:
        # Obtener correlación con la variable objetivo y quitar la de sí misma
        target_corr = corr_matrix[target].drop(target)
        
        # Obtener las 5 con mayor correlación absoluta
        top_5_corr = target_corr.abs().sort_values(ascending=False).head(5)
        
        # Recuperar los valores originales (con su respectivo signo)
        top_5_vars = target_corr[top_5_corr.index]
        
        print(f"\n--- Top 5 variables con mayor correlación con {target} ---")
        for var, val in top_5_vars.items():
            print(f"{var}: {val:.4f}")
    else:
        print(f"La variable {target} no se encuentra en las numéricas.")

    # 3. Visualizaciones
    # Configurar estilo de seaborn
    sns.set_theme(style="whitegrid")
    
    # a. Heatmap de correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title("Mapa de Calor de Correlaciones (Pearson)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'heatmap_correlacion.png'))
    plt.close()
    
    # b. Boxplot de diet_type vs weight_change_kg_6m
    if 'diet_type' in df.columns and target in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='diet_type', y=target, data=df,hue='diet_type',legend=False, palette='Set2')
        plt.title('Cambio de Peso por Tipo de Dieta', fontsize=14)
        plt.xlabel('Tipo de Dieta')
        plt.ylabel('Cambio de Peso a los 6 meses (kg)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'boxplot_dietas.png'))
        plt.close()
        
    # c. Gráfico de barras de weight_change_kg_6m agrupado por approach
    if 'approach' in df.columns and target in df.columns:
        plt.figure(figsize=(10, 6))
        # Calculamos el promedio
        avg_approach = df.groupby('approach')[target].mean().reset_index()
        # Ordenamos de menor a mayor cambio de peso para mejor lectura
        avg_approach = avg_approach.sort_values(by=target)
        
        sns.barplot(x='approach', y=target, data=avg_approach,hue='approach',legend=False, palette='viridis')
        plt.title('Promedio de Cambio de Peso por Enfoque del Nutricionista', fontsize=14)
        plt.xlabel('Enfoque del Nutricionista')
        plt.ylabel('Cambio de Peso Medio (kg)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'barras_nutricionista.png'))
        plt.close()
        
    # d. Boxplot de sex vs weight_change_kg_6m
    if 'sex' in df.columns and target in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='sex', y=target, data=df,hue='sex',legend=False, palette='Pastel1')
        plt.title('Cambio de Peso por Sexo del Paciente', fontsize=14)
        plt.xlabel('Sexo')
        plt.ylabel('Cambio de Peso a los 6 meses (kg)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'boxplot_sexo.png'))
        plt.close()
        
    print(f"\nVisualizaciones guardadas exitosamente en la carpeta: {vis_dir}")

    # 4. Respuestas en Consola
    print("\n--- Resumen del Análisis ---")
    
    # a. Dieta con mayor pérdida media
    if 'diet_type' in df.columns and target in df.columns:
        avg_diet = df.groupby('diet_type')[target].mean()
        # Asumiendo que "mayor pérdida" significa el valor más negativo
        best_diet = avg_diet.idxmin()
        best_diet_val = avg_diet.min()
        print(f"1. ¿Tipo de dieta con mayor pérdida media de peso?")
        print(f"   La dieta '{best_diet}' con una media de {best_diet_val:.2f} kg.")
        
    # b. Enfoque con mejores resultados
    if 'approach' in df.columns and target in df.columns:
        avg_app = df.groupby('approach')[target].mean()
        best_app = avg_app.idxmin()
        best_app_val = avg_app.min()
        print(f"2. ¿Enfoque del nutricionista asociado a mejores resultados (mayor pérdida)?")
        print(f"   El enfoque '{best_app}' con una media de {best_app_val:.2f} kg.")
        
    # c. Diferencia notable entre sexos
    if 'sex' in df.columns and target in df.columns:
        avg_sex = df.groupby('sex')[target].mean()
        print(f"3. ¿Hay alguna diferencia notable entre sexos?")
        for s, val in avg_sex.items():
            print(f"   - Sexo {s}: Media de cambio de peso = {val:.2f} kg")
        
        diff_max = avg_sex.max() - avg_sex.min()
        print(f"   (Diferencia máxima observada entre sexos: {diff_max:.2f} kg)")

if __name__ == "__main__":
    main()
