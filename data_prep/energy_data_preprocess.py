import pandas as pd
import os

# Define the resources directory
resources_dir = 'resources'

# Get all CSV files from the resources directory
csv_files = [f for f in os.listdir(resources_dir) if f.endswith('.csv')]

# Check if any CSV files exist
if not csv_files:
    print(f"Error: No CSV files found in the '{resources_dir}' directory")
    exit(1)

print(f"Found {len(csv_files)} CSV files in '{resources_dir}':")
for f in csv_files:
    print(f"- {f}")

try:
    # Read all CSV files into a list of dataframes
    data_frames = []
    required_columns = {'Year', 'Indicator Name', 'Value'}
    
    for csv_file in csv_files:
        file_path = os.path.join(resources_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Check for required columns
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            print(f"Error: {csv_file} is missing required columns: {', '.join(missing_cols)}")
            exit(1)
            
        data_frames.append(df)
    
    # Combine all dataframes
    if not data_frames:
        print("Error: No valid data to process")
        exit(1)
    
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # Ensure Value column is numeric, coerce errors to NaN
    df_all['Value'] = pd.to_numeric(df_all['Value'], errors='coerce')
    
    # Pivot using sum as the aggregation function
    df_wide = df_all.pivot_table(
        index='Year',
        columns='Indicator Name',
        values='Value',
        aggfunc='sum'  # Use sum instead of mean to handle potential non-numeric values
    ).reset_index()
    
    # Create data directory if it doesn't exist
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'processed_energy_data.csv')
    df_wide.to_csv(output_file, index=False)
    
    # Show results
    print("\nPreview of processed data:")
    print(df_wide.head())
    print(f"\nData successfully saved to: {os.path.abspath(output_file)}")
    
except pd.errors.EmptyDataError:
    print("Error: One or more files are empty.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)
