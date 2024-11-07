#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import sys
from geopy.geocoders import Nominatim
import numpy as np
import math
import os

# Importing the read_csv_file function from HP_Functions module
from HP_Functions import read_csv_file

class FeatureEngg:
    """
    A class to handle various operations for house price prediction including
    data loading, feature engineering, and merging datasets.

    Attributes:
        geolocator (Nominatim): Geocoder for converting addresses into geographic coordinates.
    """
    
    def __init__(self):
        """
        Initializes the FeatureEngg class with a base path.
        """
        # self.geolocator = Nominatim(user_agent="geoapiExercises")

    def get_coordinates(self, location):
        """
        Retrieves the latitude and longitude for a given location.

        Args:
            location (str): The location to geocode.

        Returns:
            tuple: A tuple containing latitude and longitude.
        """
        """ location_obj = self.geolocator.geocode(location)
        if location_obj:
             return location_obj.latitude, location_obj.longitude
         else:
             return None, None
        pass  """

    def load_and_geocode_data(self, input_file_name, output_file_name):
        
        """
        Loads a CSV file, geocodes the addresses, and saves the results to a new CSV file.
        
        Addition of new attributes: Nearest Station and Distance between the postcode and Nearest Stations
        Utilization of the NS lookup file to find the co-ordinates of post codes and nearest stations

        Args:
            input_file_name (str): The name of the input CSV file.
            output_file_name (str): The name of the output CSV file.
        """
        #BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        
        """ 
        df = read_csv_file(input_file_name)
        df['postcode_latitude'], df['postcode_longitude'] = zip(*df['postal_code'].apply(lambda x: self.get_coordinates(x)))
        df['station_latitude'], df['station_longitude'] = zip(*df['NearestStation'].apply(lambda x: self.get_coordinates(x)))
        output_file = os.path.join(BASE_PATH, output_file_name)
        df.to_csv(output_file, index=False) 
        """

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the distance between two geographic coordinates using the Haversine formula.

        Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the second point.

        Returns:
            float: Distance between the two points in miles.
        """
        """ lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        R = 3958.8  # Radius of the Earth in miles
        return R * c """

    def merge_and_calculate_distance(self, housing_file_name, coordinates_file_name, output_file_name):
        """
        Merges housing data with coordinates, calculates the distance to the nearest station, and saves the result.
        Joins the preprocessed housing file and loc_coordinates file 

        Args:
            housing_file_name (str): Name of the housing data CSV file.
            coordinates_file_name (str): Name of the coordinates data CSV file.
            output_file_name (str): Name of the output CSV file.
        """
        #BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        
        """
        df_housing = read_csv_file(housing_file_name)
        df_coordinates = read_csv_file(coordinates_file_name)
        df_merged = pd.merge(df_housing, df_coordinates[['NearestStation', 'postal_code', 'postcode_latitude', 'postcode_longitude', 'station_latitude', 'station_longitude']], on='postal_code', how='left')
        df_merged['price_per_sqft'] = df_merged['price'] / df_merged['area_in_sq_ft']
        df_merged.rename(columns={'NearestStation': 'nearest_station'}, inplace=True)
        df_merged['distance_miles'] = df_merged.apply(lambda row: round(self.haversine_distance(row['postcode_latitude'], row['postcode_longitude'], row['station_latitude'], row['station_longitude']), 2), axis=1)
        output_file = os.path.join(BASE_PATH, output_file_name)
        df_merged.to_csv(output_file, index=False) """

    def add_crime_rate_factor(self, housing_file_name, crime_file_name, output_file_name):
        """
        Merges housing data with crime rate data and saves the result.

        Args:
            housing_file_name (str): Name of the housing data CSV file.
            crime_file_name (str): Name of the crime rate data CSV file.
            output_file_name (str): Name of the output CSV file.
        """
        BASE_PATH = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        housing_df = read_csv_file(housing_file_name)
        crime_df = read_csv_file(crime_file_name)
        merged_df = pd.merge(housing_df, crime_df, left_on='location', right_on='location', how='left')
        output_file = os.path.join(BASE_PATH, output_file_name)
        merged_df.to_csv(output_file, index=False)
        print("The feature engineering file has been exported and can be used to predict house prices with machine learning algorithms")
        print("The crime rate factor has been added to the file to check the external factors affecting the house prices")

if __name__ == "__main__":
    FeatEngg = FeatureEngg()
    
    # Load and geocode data
    input_file_name = 'lkp_pc_nearest_station.csv'
    output_file_name = 'temp_pc_ns_coordinates.csv'
    FeatEngg.load_and_geocode_data(input_file_name, output_file_name)

    # Merge and calculate distance
    housing_file_name = 'temp_housing_london_loc.csv'
    coordinates_file_name = 'temp_pc_ns_coordinates.csv'
    output_file_merged_name = 'london_housing_with_stations.csv'
    FeatEngg.merge_and_calculate_distance(housing_file_name, coordinates_file_name, output_file_merged_name)

    # Add crime rate factor
    housing_with_stations_file_name = 'london_housing_with_stations.csv'
    crime_file_name = 'lkp_crimerate_influence.csv'
    final_output_file_name = 'london_housing_with_stations_crimerate.csv'
    FeatEngg.add_crime_rate_factor(housing_with_stations_file_name, crime_file_name, final_output_file_name)

    

