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

    # Validar que exista la columna sex
    if 'sex' not in df.columns:
        print("Error: La columna 'sex' no existe en el DataFrame.")
        return

    print("Valores únicos ANTES de la limpieza:", df['sex'].unique())

    # 1. Convierte toda la columna a texto (string)
    df['sex'] = df['sex'].astype(str)
    
    # 2. Elimina los espacios en blanco al principio y al final
    df['sex'] = df['sex'].str.strip()
    
    # 3. Convierte todo a mayúsculas
    df['sex'] = df['sex'].str.upper()
    
    # 4. Reemplaza explícitamente 'FEMALE' por 'F' y 'MALE' por 'M'
    df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})

    # 5. Imprime los valores únicos de la columna sex para confirmar
    print("Valores únicos DESPUÉS de la limpieza:", df['sex'].unique())

    # 6. Guarda el DataFrame modificado sobreescribiendo el archivo original
    df.to_csv(data_path, index=False)
    print("¡Archivo cleaned_data.csv limpiado y sobreescrito exitosamente!")

if __name__ == "__main__":
    main()
