# ###
# Modify the format of CSV file to be the inpur of PCA and PLS analysis
# Author: Chaofan Yin
# Date: 2024-04-18
# Note: 
#     This is part of Supporting Informatioin for the Year 3 Project Report 
#     at the University of Manchester, Department of Chemistry. 
# ###

import pandas as pd
import os

# The input file was exported from MestReNova, without labelling class of samples. 

def modify_csv(csv_file, template_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Delete the first 2 columns
    df = df.iloc[:, 2:]

    # Read the first 3 columns of the template CSV file excluding the header
    template_df = pd.read_csv(template_file, usecols=range(3), header=0)

    # Concatenate the template columns with the modified DataFrame
    df = pd.concat([template_df, df], axis=1)
    
    # Store the original header
    original_header = df.columns.tolist()  

    # Transpose the DataFrame
    df_transposed = df.T

    # Insert the original header as the first column of the transposed DataFrame
    df_transposed.insert(0, 'Original Header', original_header)

    # Write the transposed DataFrame to the CSV file
    df_transposed.to_csv(csv_file, header=False, index=False)

def process_csv_files(directory, template_file):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename != template_file:
            # Full path of the CSV file
            csv_file = os.path.join(directory, filename)
            # Apply modify_csv function to the current CSV file
            modify_csv(csv_file, template_file)


# Example usage
#csv_file = 'original_0.02_peak.csv'
template_file = 'Template.csv'
directory = 'directory/to/csv'
process_csv_files(directory, template_file)
