import re
import numpy as np
import pandas as pd
import os

def compute_average(group_name, df):
    group_columns = [col for col in df.columns if group_name in col]
    return df[group_columns].mean(axis=1)

def consolidate(freq_min, freq_max, brainwave, userdf, df_temp):
    
    pattern = re.compile(rf".*[{freq_min}-{freq_max}]Hz.*")
    columns_to_include = [col for col in userdf.columns if re.match(pattern, col)]

    bw_average = userdf[columns_to_include].mean(axis=1)
    bw_median = np.array([np.median(bw_average)])

    df_temp[f"{brainwave}"] = bw_median
    
    df_temp = pd.DataFrame()
    
folder_path = "data_samples"

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Load the CSV file into a DataFrame
        file_path = os.path.join(folder_path, filename)
        brain_data = pd.read_csv(file_path)

    df_temp = pd.DataFrame()
    df = pd.DataFrame()

    pattern = re.compile("^Aux|^f_")
    unwanted = [col for col in brain_data.columns if re.match(pattern, col)]
    userdf = brain_data.drop(columns=unwanted)
    userdf = userdf.drop('info', axis=1)

    consolidate(1, 3, "Delta", userdf, df_temp)
    consolidate(4, 7, "Theta", userdf, df_temp)
    consolidate(8, 9, "Alpha1", userdf, df_temp)
    consolidate(10, 11, "Alpha2", userdf, df_temp)
    consolidate(12, 20, "Beta1", userdf, df_temp)
    consolidate(20, 29, "Beta2", userdf, df_temp)

    # combine alpha1 + alpha2, and beta1 + beta2
    groups = ['Alpha', 'Beta']
    for group in groups:
        df_temp[f'{group}'] = compute_average(group, df_temp)

    userdf = df_temp
    userdf = userdf.drop(columns=['Alpha1', 'Alpha2', 'Beta1', 'Beta2'])
    
    output_file = os.path.join("preprocessed_data", "preprocessed_data.csv")
    userdf.to_csv(output_file, index=False)
    print(f"{filename} data saved to {output_file}")

# combine to mega csv
for filename in os.listdir("preprocessed_data"):
    if filename.endswith(".csv"):
        # Load each CSV file into a DataFrame
        file_path = os.path.join("preprocessed_data", filename)
        df = pd.read_csv(file_path)
        # Concatenate the current DataFrame to the mega DataFrame
        mega_df = pd.concat([mega_df, df], ignore_index=True)

mega_df.to_csv("mega_preprocessed_data.csv", index=False)

    