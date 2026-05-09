import os
import pandas as pd


#importacion de los csv's con rutas absolutas simples. Si cambian esto a robustas avisem pls
pacientes = pd.read_csv('../data/patients.csv')
dietas = pd.read_csv('.../data/diets.csv')
nutris = pd.read_csv('.../data/nutricionist.csv')
outcomes = pd.read_csv('.../outcomes.csv')

#merge's de los csv para tenerlo todo juntos
merge_f =  pd.merge(outcomes,pacientes,on='patient_id',how = 'left')
merge_f = pd.merge(merge_f, dietas, on='diets_id', how = 'left')
merge_f = pd.merge(merge_f, nutris, on='nutritionist_id', how='left')


#print de que se completo (Prueba mia, no se si lo dejemos)
print(f'Merge Completed: Dataset saved in data/processed_data.csv')

#tratamiento de datos (redundancias del enunciados)
colm_to_dlt =  ['bmi_redundant', 'experience_years']
merge_f.drop(columns=[c for c in colm_to_dlt if c in merge_f.columns], inplace=True)

#Tratamiento de datos (Invisibles o nulos)
merge_f['years_experience'] = merge_f['years_experience'].fillna(merge_f['years_experience'].median())
merge_f['approach'] = merge_f['approach'].fillna('unknown')
merge_f['motivitation_score'] = merge_f['motivitation_score'].fillna(merge_f['motivitation_score'].mean())

#DETECCION DE AUTLIERS (con una media, si cambian esto avisar pls)
merge_f.loc[merge_f['years_experience'] > 60, 'years_experience'] = merge_f['years_experience'].median()