#!/usr/bin/env python
# coding: utf-8

# ## Accuracy calculation: Measure R2

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Import the read_csv_file function from HP_Functions module
from HP_Functions import read_csv_file

class ExternalFactorInfAccuracy:
    """
    A class used to represent a pipeline for predicting house prices based on external factors.

    Attributes
    ----------
    file_path : str
        The file path of the dataset.
    df : DataFrame
        The dataframe containing the dataset.
    pipeline : Pipeline
        The pipeline used for preprocessing and modeling.
    X_train : array
        The training features.
    X_test : array
        The testing features.
    y_train : array
        The training target variable.
    y_test : array
        The testing target variable.

    Methods
    -------
    load_data():
        Loads the CSV file into a pandas DataFrame using the read_csv_file function.
    
    preprocess_data():
        Preprocesses the data by encoding categorical variables, scaling numerical features, 
        and splitting the data into training and test sets.
    
    train_and_evaluate():
        Trains the Linear Regression model and evaluates it on the test data using multiple metrics.
    
    predict(features):
        Predicts house prices based on the provided feature values using the trained model.
    
    run(file_path):
        Class method that runs the end-to-end process: loading, preprocessing, training, and evaluation.
    """
    
    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the ExternalFactorInfAccuracy object.
        
        Parameters
        ----------
        file_path : str
            The file path of the dataset.
        """
        self.file_path = file_path
        self.df = None
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame using the read_csv_file function.
        """
        try:
            self.df = read_csv_file(self.file_path)
            print(f"Number of records: {len(self.df)}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def preprocess_data(self):
        """
        Preprocesses the data by splitting into features and target variable and setting up the pipeline.
        """
        # Drop rows where 'price' is missing
        if 'price' in self.df.columns:
            self.df = self.df.dropna(subset=['price'])
        
        # Define the feature set and target variable
        X = self.df.drop(columns=['price', 'postcode_latitude', 'postcode_longitude', 'station_latitude', 'station_longitude'])
        y = self.df['price']

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns

        # Define the preprocessor with SimpleImputer for missing values, OneHotEncoder for categorical features,
        # and StandardScaler for numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_cols)
            ]
        )

        # Create a pipeline with preprocessing and the Linear Regression model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.385, random_state=42)

    def train_and_evaluate(self):
        """
        Trains the Linear Regression model and evaluates it on the test data using multiple metrics.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not set up. Call preprocess_data() before training.")
        
        # Fit the pipeline on the training data
        self.pipeline.fit(self.X_train, self.y_train)

        # Predict on the test set
        y_pred_lr = self.pipeline.predict(self.X_test)

        # Calculate metrics
        r2_lr = r2_score(self.y_test, y_pred_lr)
        mae = mean_absolute_error(self.y_test, y_pred_lr)
        mse = mean_squared_error(self.y_test, y_pred_lr)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((self.y_test - y_pred_lr) / self.y_test)) * 100

        # Print the metrics
        print(f"Linear Regression R-squared: {round(r2_lr, 2)}")
        print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
        print(f"Mean Absolute Percentage Error (MAPE): {round(mape, 2)}%")

    def predict(self, features):
        """
        Predicts house prices based on the provided feature values using the trained model.
        
        Parameters
        ----------
        features : dict
            A dictionary containing feature values for prediction.

        Returns
        -------
        float
            The predicted price.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not set up. Call preprocess_data() before predicting.")
        
        # Convert features dictionary to DataFrame
        features_df = pd.DataFrame([features])
        
        # Predict using the trained pipeline
        predicted_price = self.pipeline.predict(features_df)
        
        return predicted_price[0]

    @classmethod
    def run(cls, file_path):
        """
        Runs the end-to-end process: loading, preprocessing, training, and evaluation.
        
        Parameters
        ----------
        file_path : str
            The file path of the dataset.
        """
        # Create an instance of ExternalFactorInfAccuracy
        predictor = cls(file_path)

        # Execute the methods
        predictor.load_data()
        predictor.preprocess_data()
        predictor.train_and_evaluate()
        
        # Example of predicting with new data
        new_data = {
            'house_type': 'Penthouse',
            'area_in_sq_ft': 2716,
            'no_of_bedrooms': 5,
            'no_of_bathrooms': 5,
            'no_of_receptions': 5,
            'distance_miles': 0.71,
            'no_of_crimes_per_year': 28092,
            'price_per_sqft': 61.67
        }
        
        predicted_price = predictor.predict(new_data)
        print(f"\nUsing the dataset with various property features, the model predicts prices based on the given parameters.")
        print(f"For a property with an area of 2716 sqft, a distance of 0.71 miles, and 28092 crimes per year,")
        print(f"the predicted price is: £{predicted_price:.2f}.")
        print("\nYou can use the `predict` function in the code by passing these significant parameters to get the predicted price.")
         
        return predictor

# Execute the class method
if __name__ == "__main__":
    file_path = r"london_housing_with_stations_crimerate.csv"
    predictor = ExternalFactorInfAccuracy.run(file_path)

