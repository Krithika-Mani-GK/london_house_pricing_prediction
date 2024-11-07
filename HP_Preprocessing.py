#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

"""
Module for preprocessing house pricing data.
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display, HTML
from HP_Functions import read_csv_file


class HousePricingPreprocessor:
    """
    A class to preprocess house pricing data.
    """
    
    def __init__(self, house_pricing_file_name, lookup_file_name):
        """
        Initialize the preprocessor with file names.
        
        Parameters:
        house_pricing_file_name (str): Name of the house pricing data file.
        lookup_file_name (str): Name of the lookup data file.
        """
        self.house_pricing_file_name = house_pricing_file_name
        self.lookup_file_name = lookup_file_name
        self.base_path = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        self.house_pricing_df = read_csv_file(self.house_pricing_file_name)
        self.lookup_df = pd.read_csv(os.path.join(self.base_path, lookup_file_name), encoding='latin-1')
        self.processed_df = None
        self.scaler = MinMaxScaler()  # Initialize the scaler here
        self.scaler_fitted = False

    def preprocess_house_pricing_data(self):
        """
        Preprocess the house pricing data.
        
        Returns:
        pd.DataFrame: Preprocessed DataFrame.
        """
        df = self.house_pricing_df.copy()
        
        # Rename and drop columns
        df.rename(columns={df.columns[0]: "Number"}, inplace=True)
        df.drop(columns=["Number"], inplace=True)
        df.rename(columns={
            "Property Name": "property_name",
            "Price": "price",
            "House Type": "house_type",
            "Area in sq ft": "area_in_sq_ft",
            "No. of Bedrooms": "no_of_bedrooms",
            "No. of Bathrooms": "no_of_bathrooms",
            "No. of Receptions": "no_of_receptions",
            "Location": "location",
            "City/County": "city",
            "Postal Code": "postal_code"
        }, inplace=True)
        
        # Filter rows where city is London
        df = df[df["city"].str.lower() == "london"]
        
        self.processed_df = df
        return df

    def validate_base_file(self):
        """
        Perform validation on the preprocessed data.
        
        Returns:
        tuple: Total number of attributes, record count, DataFrame.
        """
        total_attributes = len(self.processed_df.columns)
        record_count = len(self.processed_df)
        
        return total_attributes, record_count, self.processed_df.head()
    
    def validate_lookup_file(self):
        """
        Perform validation on the lookup data.
        
        Returns:
        tuple: Record count, duplicate postal codes, DataFrame.
        """
        postal_code_counts = self.lookup_df['postal_code'].value_counts()
        duplicate_postal_codes = postal_code_counts[postal_code_counts > 1]
        
        return len(self.lookup_df), duplicate_postal_codes, self.lookup_df.head()
    
    def handle_missing_values(self):
        """
        Handle missing values by merging with the lookup DataFrame.
        
        Returns:
        pd.DataFrame: Merged DataFrame.
        """
        merged_df = self.processed_df.merge(self.lookup_df, on="postal_code", how="left")
        merged_df.drop(columns=['location_x'], inplace=True)
        merged_df.rename(columns={'location_y': 'location'}, inplace=True)
        
        self.processed_df = merged_df
        return merged_df
    
    def validate_merged_file(self):
        """
        Perform validation on the merged data.
        
        Returns:
        tuple: Rows with missing location, count of attributes, count of missing values, DataFrame.
        """
        filtered_df = self.processed_df[self.processed_df['location'].isna()]
        isnull_sum = self.processed_df.isnull().sum()
        
        return filtered_df, len(self.processed_df.columns), isnull_sum, self.processed_df.head()
    
    def analyze_location_greater_than_avg_price(self, average_price):
        """
        Analyze locations where the average price is greater than the specified value.
        
        Parameters:
        average_price (float): Price threshold for filtering.
        
        Returns:
        pd.Series: Count of records for each location.
        """
        grouped_df = self.processed_df.groupby('location').filter(lambda x: x['price'].mean() > average_price)
        location_counts = grouped_df['location'].value_counts()
        
        return location_counts
    
    def handle_duplicates(self):
        """
        Handle duplicate records in the DataFrame.
        
        Returns:
        int: Number of duplicate records after removal.
        """
        self.processed_df.drop_duplicates(inplace=True)
        return self.processed_df.duplicated().sum()
    
    def one_hot_encode(self, column_name):
        """
        One-hot encode the specified column.
        
        Parameters:
        column_name (str): The name of the column to encode.
        
        Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        one_hot_encoded = pd.get_dummies(self.processed_df[column_name], prefix=column_name)
        self.processed_df = pd.concat([self.processed_df, one_hot_encoded], axis=1)
        return self.processed_df
    
    def remove_one_hot_columns(self, column_name):
        """
        Remove one-hot encoded columns from the DataFrame.
        
        Parameters:
        column_name (str): The name of the column whose one-hot encoded columns will be removed.
        
        Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns removed.
        """
        one_hot_columns = [col for col in self.processed_df.columns if col.startswith(f'{column_name}_')]
        self.processed_df.drop(columns=one_hot_columns, inplace=True)
        return self.processed_df
    
    def remove_outliers(self, column_name):
        """
        Remove outliers from the specified column using IQR method.
        
        Parameters:
        column_name (str): The name of the column from which to remove outliers.
        
        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        Q1 = self.processed_df[column_name].quantile(0.25)
        Q3 = self.processed_df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        self.processed_df = self.processed_df[(self.processed_df[column_name] > (Q1 - 1.5 * IQR)) & 
                                              (self.processed_df[column_name] < (Q3 + 1.5 * IQR))]
        return self.processed_df
    
    def normalize_column(self, column_name):
        """
        Normalize the specified column using Min-Max scaling.
        
        Parameters:
        column_name (str): The name of the column to normalize.
        
        Returns:
        pd.DataFrame: DataFrame with normalized column.
        """
        self.processed_df[f'{column_name}_normalized'] = self.scaler.fit_transform(self.processed_df[[column_name]])
        self.scaler_fitted = True  # Mark scaler as fitted
        return self.processed_df
    
    def denormalize_column(self, column_name):
        """
        De-normalize the specified column to its original values.
        
        Parameters:
        column_name (str): The name of the column to denormalize.
        
        Returns:
        pd.DataFrame: DataFrame with denormalized column.
        """
        if not self.scaler_fitted:
            raise RuntimeError("Scaler has not been fitted. Call 'normalize_column' first.")
        
        self.processed_df[f'{column_name}_original'] = self.scaler.inverse_transform(self.processed_df[[f'{column_name}_normalized']])
        
        # Drop the normalized column
        self.processed_df.drop(columns=[f'{column_name}_normalized', f'{column_name}_original'], inplace=True)
        return self.processed_df
    
    def export_to_csv(self, output_file_name):
        """
        Export the processed DataFrame to a CSV file.
        
        Parameters:
        output_file_name (str): The name of the output CSV file.
        """
        output_file_path = os.path.join(self.base_path, output_file_name)
        self.processed_df.to_csv(output_file_path, index=False)
        print(f"The preprocessed file has been exported to {output_file_path}.")
    
    def display_records(self):
        """
        Display the records of the processed DataFrame.
        """
        display(HTML("<b>Displaying records of the processed DataFrame:</b>"))
        display(self.processed_df)  # Display as a table

if __name__ == "__main__":
    house_pricing_file_name = 'housing_london_dataset.csv'
    lookup_file_name = 'lkp_pc_loc.csv'
    output_file_name = 'temp_housing_london_loc.csv'
    
    preprocessor = HousePricingPreprocessor(house_pricing_file_name, lookup_file_name)
    preprocessor.preprocess_house_pricing_data()
    preprocessor.handle_missing_values()
    preprocessor.handle_duplicates()
    preprocessor.one_hot_encode('house_type')
    preprocessor.remove_one_hot_columns('house_type')  # Remove one-hot columns
    preprocessor.remove_outliers('price')
    preprocessor.normalize_column('price')
    preprocessor.denormalize_column('price')
    
    preprocessor.export_to_csv(output_file_name)
    
    # Display records after all processing
    preprocessor.display_records()

