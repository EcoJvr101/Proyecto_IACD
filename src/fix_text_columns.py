import pandas as pd
import os

def main():
    # Definir la ruta exacta de cleaned_data.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_data.csv')
    
    print(f"Cargando archivo: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {data_path}")
        return

    # Seleccionar dinámicamente columnas de tipo texto (object/string) excluyendo 'sex'
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'sex' in text_cols:
        text_cols.remove('sex')
        
    print(f"Columnas detectadas para limpieza: {text_cols}")
    
    # Aplicar transformaciones a todas las columnas de texto seleccionadas
    for col in text_cols:
        # 1. Asegurar tipo string (protección frente a nulos o mezclas)
        df[col] = df[col].astype(str)
        # 2. Eliminar espacios en blanco al principio y al final
        df[col] = df[col].str.strip()
        # 3. Capitalizar (primera en mayúscula, el resto en minúscula)
        df[col] = df[col].str.capitalize()
        
    # Imprimir valores únicos de la columna 'approach' para confirmar la corrección
    if 'approach' in df.columns:
        print("\nValores únicos en 'approach' DESPUÉS de la limpieza:")
        print(df['approach'].unique())
    else:
        print("\nAdvertencia: La columna 'approach' no se encontró en el DataFrame.")
        
    # Guarda el DataFrame modificado sobreescribiendo el archivo original
    df.to_csv(data_path, index=False)
    print("\n¡Archivo cleaned_data.csv limpiado masivamente y sobreescrito exitosamente!")

if __name__ == "__main__":
    main()
