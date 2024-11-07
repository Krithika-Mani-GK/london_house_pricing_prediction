#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate 
from HP_Functions import read_csv_file
from ydata_profiling import ProfileReport

class HousePriceEDA:
    """
    A class used to perform exploratory data analysis (EDA) on house pricing data.

    Attributes
    ----------
    csv_path : str
        The file path of the dataset.
    house_pricing_df : DataFrame
        The dataframe containing the cleaned house pricing data.

    Methods
    -------
    load_and_clean_data():
        Loads the dataset from the CSV file and performs basic data cleaning.
    
    describe_data():
        Prints the statistical summary of the dataset.
    
    plot_price_distribution():
        Plots the distribution of house prices.
    
    plot_location_count():
        Plots the count of records for locations where the average house price is greater than 1,000,000.
    
    plot_house_type_count():
        Plots the count of different house types.
    
    plot_missing_values_heatmap():
        Plots a heatmap of missing values in the dataset.
    
    plot_bedroom_bathroom_reception_distribution():
        Plots the distribution of the number of bedrooms, bathrooms, and receptions.
    
    plot_top_bottom_location_prices():
        Plots the price distribution for the top 10 and bottom 10 locations.
    
    plot_correlation_matrix():
        Plots the correlation matrix of numerical features.
    
    generate_profile_report():
        Generates a detailed profiling report of the dataset.
    
    plot_per_column_distribution(n_graphs_shown, n_graphs_per_row):
        Plots the distribution of each column in the dataset.
    
    univariate_bivariate_analysis():
        Performs univariate and bivariate analysis on the dataset.
    
    perform_eda():
        Executes the full EDA process including all the plots and analyses.
    """
    
    def __init__(self, csv_path):
        """
        Constructs all the necessary attributes for the HousePriceEDA object.

        Parameters
        ----------
        csv_path : str
            The file path of the dataset.
        """
        self.csv_path = csv_path
        self.house_pricing_df = self.load_and_clean_data()

    def load_and_clean_data(self):
        """
        Loads the dataset from the CSV file and performs basic data cleaning.

        Returns
        -------
        DataFrame
            The cleaned house pricing data.
        """
        # Load the dataset
        house_pricing_df = read_csv_file(self.csv_path)
        
        # Basic cleansing
        house_pricing_df.rename(columns={house_pricing_df.columns[0]: "Number"}, inplace=True)
        house_pricing_df.drop(columns=["Number"], inplace=True)
        house_pricing_df = house_pricing_df[house_pricing_df["City/County"].str.lower() == "london"]

        # Convert inf values to NaN
        house_pricing_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

        return house_pricing_df

    def describe_data(self):
        """
        Prints the statistical summary of the dataset.
        """
        description = self.house_pricing_df.describe()
        print(tabulate(description, headers='keys', tablefmt='pretty'))
        
        #print(self.house_pricing_df.describe())

    def plot_with_heading(self, title):
        """
        Helper function to add a heading before each plot.

        Parameters
        ----------
        title : str
            The title of the plot.
        """
        plt.figure(figsize=(10, 1))  # Adjusting the figure size for the heading
        plt.text(0.5, 0.5, title, ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.axis('off')
        plt.show()

    def plot_price_distribution(self):
        """
        Plots the distribution of house prices.
        """
        self.plot_with_heading('Distribution of House Prices')
        plt.figure(figsize=(8, 5))
        sns.histplot(data=self.house_pricing_df, x='Price', bins=30, kde=True)
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

    def plot_location_count(self):
        """
        Plots the count of records for locations where the average house price is greater than 1,000,000.
        """
        self.plot_with_heading('Count of Records for Locations with Average Price > 1,000,000')
        filtered_df = self.house_pricing_df[self.house_pricing_df['Price'] > 1000000]
        location_counts = filtered_df['Location'].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=location_counts.index, y=location_counts.values, palette='viridis')
        plt.xlabel('Location')
        plt.ylabel('Count of Records')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_house_type_count(self):
        """
        Plots the count of different house types.
        """
        self.plot_with_heading('Count of House Types')
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.house_pricing_df, x='House Type')
        plt.xlabel('House Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    def plot_missing_values_heatmap(self):
        """
        Plots a heatmap of missing values in the dataset.
        """
        self.plot_with_heading('Heatmap of Missing Values')
        plt.figure(figsize=(8, 5))
        sns.heatmap(self.house_pricing_df.isnull(), cbar=False, cmap='viridis')
        plt.show()

    def plot_bedroom_bathroom_reception_distribution(self):
        """
        Plots the distribution of the number of bedrooms, bathrooms, and receptions.
        """
        self.plot_with_heading('Distribution of Number of Bedrooms, Bathrooms, and Receptions')
        fig, axes = plt.subplots(1, 3, figsize=(8, 5))
        sns.countplot(ax=axes[0], data=self.house_pricing_df, x='No. of Bedrooms')
        axes[0].set_title('Distribution of Number of Bedrooms')
        sns.countplot(ax=axes[1], data=self.house_pricing_df, x='No. of Bathrooms')
        axes[1].set_title('Distribution of Number of Bathrooms')
        sns.countplot(ax=axes[2], data=self.house_pricing_df, x='No. of Receptions')
        axes[2].set_title('Distribution of Number of Receptions')
        plt.tight_layout()
        plt.show()

    def plot_top_bottom_location_prices(self):
        """
        Plots the price distribution for the top 10 and bottom 10 locations.
        """
        self.plot_with_heading('Distribution of Prices for Top 10 and Bottom 10 Locations')
        top_10_locations = self.house_pricing_df['Location'].value_counts().head(10).index
        top_10_df = self.house_pricing_df[self.house_pricing_df['Location'].isin(top_10_locations)]
        bottom_10_locations = self.house_pricing_df['Location'].value_counts().tail(10).index
        bottom_10_df = self.house_pricing_df[self.house_pricing_df['Location'].isin(bottom_10_locations)]
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        sns.boxplot(ax=axes[0], data=top_10_df, x='Location', y='Price', order=top_10_locations)
        axes[0].set_title('Distribution of Prices for Top 10 Locations')
        sns.boxplot(ax=axes[1], data=bottom_10_df, x='Location', y='Price', order=bottom_10_locations)
        axes[1].set_title('Distribution of Prices for Bottom 10 Locations')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of numerical features.
        """
        self.plot_with_heading('Correlation Matrix of Numerical Features')
        numerical_cols = self.house_pricing_df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = self.house_pricing_df[numerical_cols].corr()
        plt.figure(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.show()

    def generate_profile_report(self):
        """
        Generates a detailed profiling report of the dataset.
        """
        profile = ProfileReport(self.house_pricing_df, title="Pandas Profiling Report", explorative=True)
        profile.to_notebook_iframe()

    def plot_per_column_distribution(self, n_graphs_shown, n_graphs_per_row):
        """
        Plots the distribution of each column in the dataset.

        Parameters
       
        ----------
        n_graphs_shown : int
            The number of graphs to display.
        n_graphs_per_row : int
            The number of graphs to display per row.
        """
        self.plot_with_heading('Distribution of Each Column')
        num_cols = self.house_pricing_df.shape[1]
        n_rows = (num_cols + n_graphs_per_row - 1) // n_graphs_per_row
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_graphs_per_row, figsize=(20, 5 * n_rows))
        for i, col in enumerate(self.house_pricing_df.columns):
            ax = axes[i // n_graphs_per_row, i % n_graphs_per_row]
            sns.histplot(self.house_pricing_df[col], ax=ax, kde=True)
            ax.set_title(col)
        plt.tight_layout()
        plt.show()

    def univariate_bivariate_analysis(self):
        """
        Perform univariate and bivariate analysis on the dataset.
        """
        self.plot_with_heading('Univariate and Bivariate Analysis')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Create subplots for the univariate analysis
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))

            # Univariate Analysis
            # Histogram for 'Price'
            sns.histplot(self.house_pricing_df['Price'], kde=True, color='blue', ax=axes[0, 0])
            axes[0, 0].set_title('Price Distribution')
            axes[0, 0].set_xlabel('Price')
            axes[0, 0].set_ylabel('Frequency')

            # Box plot for 'Area in sq ft'
            sns.boxplot(y=self.house_pricing_df['Area in sq ft'], color='green', ax=axes[0, 1])
            axes[0, 1].set_title('Area in Sq Ft Distribution')
            axes[0, 1].set_ylabel('Area in Sq Ft')

            # Count plot for 'House Type'
            sns.countplot(x=self.house_pricing_df['House Type'], palette='viridis', ax=axes[0, 2])
            axes[0, 2].set_title('House Type Distribution')
            axes[0, 2].set_xlabel('House Type')
            axes[0, 2].set_ylabel('Count')

            # Bivariate Analysis
            # Scatter plot for 'Price' vs 'Area in sq ft'
            sns.scatterplot(x='Area in sq ft', y='Price', data=self.house_pricing_df, hue='House Type', palette='viridis', ax=axes[1, 0])
            axes[1, 0].set_title('Price vs Area in Sq Ft')
            axes[1, 0].set_xlabel('Area in Sq Ft')
            axes[1, 0].set_ylabel('Price')

            # Box plot for 'Price' vs 'House Type'
            sns.boxplot(x='House Type', y='Price', data=self.house_pricing_df, palette='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('Price vs House Type')
            axes[1, 1].set_xlabel('House Type')
            axes[1, 1].set_ylabel('Price')

            # Hide the empty subplot
            fig.delaxes(axes[1, 2])

            # Adjust layout to prevent overlapping
            plt.tight_layout()

            # Display the plots
            plt.show()

    def perform_eda(self):
        """
        Executes the full EDA process including all the plots and analyses.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            self.describe_data()
            self.plot_price_distribution()
            self.plot_location_count()
            self.plot_house_type_count()
            self.plot_missing_values_heatmap()
            self.plot_bedroom_bathroom_reception_distribution()
            self.plot_top_bottom_location_prices()
            self.plot_correlation_matrix()
            self.generate_profile_report()
            self.plot_per_column_distribution(10, 5)
            self.univariate_bivariate_analysis()

# Execute the EDA
if __name__ == "__main__":
    csv_path = "housing_london_dataset.csv"
    eda = HousePriceEDA(csv_path)
    eda.perform_eda()

