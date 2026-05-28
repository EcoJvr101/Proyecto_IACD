import pandas as pd
import numpy as np
import os

def main():
    # 1. Definir rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'merged_data.csv')
    
    try:
        df = pd.read_csv(input_path)
        print(f"Datos integrados cargados exitosamente. Dimensión original: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {input_path}. Asegúrate de ejecutar la fase de integración.")
        return

    # 2. Limpieza de Formato de Texto (Correcciones preventivas)
    
    # 2.1 Columna 'sex'
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})
        
    # 2.2 Otras columnas de texto
    text_cols = ['diet_name', 'diet_type', 'approach', 'specialty']
    for col in text_cols:
        if col in df.columns:
            # Convertimos a string, quitamos espacios extra y capitalizamos para unificar formatos
            df[col] = df[col].astype(str).str.strip().str.capitalize()

    # 3. Eliminación de redundancias
    cols_to_drop = ['bmi_redundant', 'experience_years']
    # Eliminar solo las que realmente existan en el DataFrame para evitar errores
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    
    if cols_to_drop_existing:
        df.drop(columns=cols_to_drop_existing, inplace=True)
        print(f"Columnas eliminadas por redundancia: {cols_to_drop_existing}")

    # 4. Tratamiento de Nulos (Missing Values)
    
    # 4.1 Eliminar filas con nulos en variables clave
    target_cols = ['weight_change_kg_6m', 'adherence_ratio']
    targets_present = [col for col in target_cols if col in df.columns]
    
    if targets_present:
        filas_antes = len(df)
        df.dropna(subset=targets_present, inplace=True)
        filas_eliminadas = filas_antes - len(df)
        print(f"Se eliminaron {filas_eliminadas} filas por nulos en las variables clave {targets_present}.")

    # 4.2 Imputación de valores faltantes
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Imputar numéricas con la mediana
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Imputar categóricas con la moda (el valor más frecuente)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # 5. Tratamiento de Outliers (Método IQR y Capping)
    # Se incluye weight_change_kg_6m porque presentaba valores extremos (ej. -100 kg)
    outlier_cols = ['age', 'baseline_weight_kg', 'height_cm', 'weight_change_kg_6m']
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Capping: limitar valores a los bigotes
            df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 6. Validación y Salida
    print("\n--- Validación de Limpieza de Texto ---")
    if 'sex' in df.columns:
        print("Valores únicos en 'sex':", df['sex'].unique())
    if 'approach' in df.columns:
        print("Valores únicos en 'approach':", df['approach'].unique())
        
    nulls_remaining = df.isnull().sum().sum()
    print("\n--- Resumen Final de Limpieza ---")
    print(f"Cantidad de nulos restantes: {nulls_remaining}")
    print(f"Dimensión final del DataFrame limpio: {df.shape}")
    
    # Exportar el DataFrame resultante
    output_path = os.path.join(data_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDatos limpios unificados exportados exitosamente a: {output_path}")

if __name__ == "__main__":
    main()
