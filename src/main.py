import os
import pandas as pd

# 1. RUTAS ROBUSTAS
# Esto detecta automáticamente dónde está este script (en tu caso, la carpeta 'src')
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Esto sube una carpeta y entra a 'data' de forma segura
directorio_data = os.path.join(directorio_actual, '..', 'data')

print(f"Buscando archivos en: {directorio_data}")

# 2. CARGA DE DATOS
pacientes = pd.read_csv(os.path.join(directorio_data, 'patients.csv'))
dietas = pd.read_csv(os.path.join(directorio_data, 'diets.csv'))
nutris = pd.read_csv(os.path.join(directorio_data, 'nutritionists.csv'))
outcomes = pd.read_csv(os.path.join(directorio_data, 'outcomes.csv'))

# 3. MERGE
merge_f = pd.merge(outcomes, pacientes, on='patient_id', how='left')
merge_f = pd.merge(merge_f, dietas, on='diet_id', how='left')
merge_f = pd.merge(merge_f, nutris, on='nutritionist_id', how='left')

# 4. LIMPIEZA
colm_to_dlt = ['bmi_redundant', 'experience_years']
merge_f.drop(columns=[c for c in colm_to_dlt if c in merge_f.columns], inplace=True)

merge_f['years_experience'] = merge_f['years_experience'].fillna(merge_f['years_experience'].median())
merge_f['approach'] = merge_f['approach'].fillna('unknown')
merge_f['motivation_score'] = merge_f['motivation_score'].fillna(merge_f['motivation_score'].mean())
merge_f.loc[merge_f['years_experience'] > 60, 'years_experience'] = merge_f['years_experience'].median()

# 5. GUARDADO ROBUSTO
ruta_guardado = os.path.join(directorio_data, 'processed_data.csv')
merge_f.to_csv(ruta_guardado, index=False)

print(f"¡Éxito! Archivo guardado correctamente en: {ruta_guardado}")