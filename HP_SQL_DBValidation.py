#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import sqlite3
import os

# Import the read_csv_file function from HP_Functions module
from HP_Functions import read_csv_file

class SqlDbValidation:
    """
    A class to process house price data and perform various operations 
    such as reading CSV files, cleaning data, performing SQL operations, 
    and exporting the final cleaned data to a CSV file.
    """
    
    def __init__(self, lookup_file_name, housing_file_name, output_file_name, db_name='house_price_prediction.db'):
        """
        Initialize the processor with file names and database name.

        Parameters:
        lookup_file_name (str): Name of the lookup CSV file.
        housing_file_name (str): Name of the housing data CSV file.
        output_file_name (str): Name to export the cleaned CSV file.
        db_name (str): Name of the SQLite database file.
        """
        self.lookup_file_name = lookup_file_name
        self.housing_file_name = housing_file_name
        self.output_file_name = output_file_name
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def load_data(self):
        """Load data from CSV files into pandas DataFrames."""
        BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        self.lookup_df = pd.read_csv(os.path.join(BASE_PATH, self.lookup_file_name), encoding='latin-1')
        self.housing_df = read_csv_file(self.housing_file_name)

    def clean_housing_data(self):
        """Clean the housing data by renaming columns and filtering for London."""
        self.housing_df = self.housing_df.iloc[:, 1:]  # Drop the first column
        self.housing_df.rename(columns={
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
        self.housing_df = self.housing_df[self.housing_df["city"].str.lower() == "london"]

    def setup_database(self):
        """Set up the SQLite database connection and cursor."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def drop_existing_views_and_tables(self):
        """Drop existing views and tables if they exist in the database."""
        self.cursor.execute("DROP VIEW IF EXISTS london_loc;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_filtered;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_dedup;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_final;")

    def write_data_to_sql(self):
        """Write the DataFrames to SQL tables."""
        self.lookup_df.to_sql('lookup', self.conn, if_exists='replace', index=False)
        self.housing_df.to_sql('london_housing', self.conn, if_exists='replace', index=False)

    def base_table_count_checks(self):
        """Perform base table count checks and print the counts."""
        self.cursor.execute("SELECT COUNT(*) FROM lookup;")
        print(f"Rows in 'lookup' table: {self.cursor.fetchone()[0]}")
        self.cursor.execute("SELECT COUNT(*) FROM london_housing;")
        print(f"Rows in 'london_housing' table: {self.cursor.fetchone()[0]}")

    def create_views_and_tables(self):
        """Create views and tables in the database for further processing."""
        self.cursor.execute("""
        CREATE VIEW london_loc AS
        SELECT lh.property_name, lh.price, lh.house_type, lh.area_in_sq_ft, lh.no_of_bedrooms, lh.no_of_bathrooms, lh.no_of_receptions,
        lh.city, lh.postal_code, l.location as location
        FROM london_housing lh
        LEFT JOIN lookup l ON lh.postal_code = l.postal_code;
        """)

        self.cursor.execute("""
        CREATE TABLE london_loc_dedup AS
        SELECT DISTINCT *
        FROM london_loc;
        """)

    def remove_outliers(self):
        """Remove outliers from the dataset based on price using the IQR method."""
        self.london_loc_df = pd.read_sql_query("SELECT * FROM london_loc_dedup;", self.conn)
        Q1 = self.london_loc_df['price'].quantile(0.25)
        Q3 = self.london_loc_df['price'].quantile(0.75)
        IQR = Q3 - Q1
        self.london_loc_df = self.london_loc_df[(self.london_loc_df['price'] > (Q1 - 1.5 * IQR)) & (self.london_loc_df['price'] < (Q3 + 1.5 * IQR))]
        print(f"Number of records after removing outliers and duplicates and post merging with the lkp_loc file: {len(self.london_loc_df)}")
        print("\n")

    def export_cleaned_data(self):
        """Export the cleaned DataFrame to a CSV file."""
        # Define the base path as a constant
        BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        self.london_loc_df.to_csv(os.path.join(BASE_PATH, self.output_file_name), index=False)
        print(f"The SQL validation is successful, the record counts match")
        print(f"The preprocessed file has been exported to: {self.output_file_name}")

    def cleanup_database(self):
        """Drop views and tables to clean up the database."""
        self.cursor.execute("DROP VIEW IF EXISTS london_loc;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_filtered;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_dedup;")
        self.cursor.execute("DROP TABLE IF EXISTS london_loc_final;")

    def close_connection(self):
        """Commit changes and close the database connection."""
        self.conn.commit()
        self.conn.close()

    def process(self):
        """Run the complete data processing pipeline."""
        # Define the base path as a constant
        BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"

        self.load_data()
        self.clean_housing_data()
        self.setup_database()
        self.drop_existing_views_and_tables()
        self.write_data_to_sql()
        self.base_table_count_checks()
        self.create_views_and_tables()
        self.remove_outliers()
        self.export_cleaned_data()
        self.cleanup_database()
        self.close_connection()

# Example usage:
if __name__ == "__main__":
    lookup_file_name = "lkp_pc_loc.csv"
    housing_file_name = "housing_london_dataset.csv"
    output_file_name = "temp_housing_london_loc.csv"

    processor = SqlDbValidation(lookup_file_name, housing_file_name, output_file_name)
    processor.process()

