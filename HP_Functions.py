#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
from tabulate import tabulate

def read_csv_file(file_name):
    """
    Reads the CSV file containing house pricing data in London.

    Parameters:
    file_name (str): The name of the CSV file.

    Returns:
    DataFrame: Contains the house pricing data.
    """
    
    # Define the base path as a constant
    BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"

    file_path = os.path.join(BASE_PATH, file_name)
    
    # Read the CSV file into a DataFrame
    house_pricing_df = pd.read_csv(file_path, index_col=None)
    
    return house_pricing_df

if __name__ == "__main__":
    df = read_csv_file("housing_london_dataset.csv")
    
    # Print the DataFrame in table format
    print(tabulate(df.head(), headers='keys', tablefmt='psql'))

