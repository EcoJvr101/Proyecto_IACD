import pandas as pd
import os

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    outcomes_path = os.path.join(data_dir, 'outcomes.csv')
    patients_path = os.path.join(data_dir, 'patients.csv')
    diets_path = os.path.join(data_dir, 'diets.csv')
    nutritionists_path = os.path.join(data_dir, 'nutritionists.csv')
    
    # Load the 4 CSV files
    try:
        outcomes = pd.read_csv(outcomes_path)
        patients = pd.read_csv(patients_path)
        diets = pd.read_csv(diets_path)
        nutritionists = pd.read_csv(nutritionists_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Perform left merges using outcomes as the base
    df_merged = outcomes.merge(patients, on='patient_id', how='left')
    df_merged = df_merged.merge(diets, on='diet_id', how='left')
    df_merged = df_merged.merge(nutritionists, on='nutritionist_id', how='left')
    
    # Print the shapes of the original DataFrames
    print("--- Dimensiones de los DataFrames Originales ---")
    print(f"Outcomes:      {outcomes.shape}")
    print(f"Patients:      {patients.shape}")
    print(f"Diets:         {diets.shape}")
    print(f"Nutritionists: {nutritionists.shape}")
    
    # Print the shape of the merged DataFrame
    print("\n--- Dimensión del DataFrame Integrado ---")
    print(f"df_merged:     {df_merged.shape}")
    
    # Export the resulting DataFrame to 'merged_data.csv' in the data directory
    output_path = os.path.join(data_dir, 'merged_data.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"\nDatos integrados exportados exitosamente a: {output_path}")

if __name__ == "__main__":
    main()
