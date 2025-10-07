
import os

original_file_path = os.path.join('data', 'processed_energy_data.csv')
cleaned_file_path = os.path.join('data', 'processed_energy_data_cleaned.csv')

with open(original_file_path, 'r') as infile, open(cleaned_file_path, 'w', newline='') as outfile:
    for i, line in enumerate(infile):
        if i != 1:  # Skip the second line (index 1)
            outfile.write(line)

print(f"Cleaned file saved to: {cleaned_file_path}")
