#!/usr/bin/env python
# coding: utf-8

# ## Accuracy calculation: Measure R2

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import warnings
from tabulate import tabulate
import scipy.stats as stats

# Import the read_csv_file function from HP_Functions module
from HP_Functions import read_csv_file

class ModelPipeline:
    """
    A class used to represent a machine learning model pipeline for housing price prediction.
    This class handles data loading, preprocessing, model training, and performance evaluation.
    """
    
    def __init__(self, file_name):
        """
        Initialize the ModelPipeline with the path to the data file and setup variables.
        
        Parameters:
            file_name (str): The name of the CSV file containing the housing data.
        """
        self.base_path = r"C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource"
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.r2_scores = {}
        self.mae_scores = {}
        self.mape_scores = {}
        self.file_name = file_name
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load data from a CSV file into a DataFrame.
        """
        full_path = f"{self.base_path}\\{self.file_name}"
        self.df = read_csv_file(full_path)
    
    def preprocess_data(self):
        """
        Preprocess the data by removing irrelevant columns, encoding categorical variables,
        scaling features, and splitting into training and testing sets.
        """
        # Drop columns that are not needed for model training
        self.df = self.df.drop(columns=['property_name', 'postal_code', 'nearest_station', 'city', 'location'])
        
        # Encode categorical columns
        label_encoders = {}
        for column in self.df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            label_encoders[column] = le

        # Separate features and target variable
        X = self.df.drop(columns=['price'])
        y = self.df['price']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics: R-squared, MAE, and MAPE.
        
        Parameters:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted values from the model.
        
        Returns:
            dict: Dictionary containing R-squared, MAE, and MAPE metrics.
        """
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'r2': r2, 'mae': mae, 'mape': mape}

    def train_linear_regression(self):
        """
        Train a Linear Regression model and evaluate its performance.
        """
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.calculate_metrics(self.y_test, y_pred)
        self.models['Linear Regression'] = model
        self.r2_scores['Linear Regression'] = metrics['r2']
        self.mae_scores['Linear Regression'] = metrics['mae']
        self.mape_scores['Linear Regression'] = metrics['mape']

    def train_xgboost(self):
        """
        Train an XGBoost model and evaluate its performance.
        """
        model = XGBRegressor(n_estimators=8, max_depth=2, random_state=42)
        X_train_xgb, _, y_train_xgb, _ = train_test_split(self.X_train, self.y_train, test_size=0.5, random_state=42)
        model.fit(X_train_xgb, y_train_xgb)
        y_pred = model.predict(self.X_test)
        metrics = self.calculate_metrics(self.y_test, y_pred)
        self.models['XGBoost'] = model
        self.r2_scores['XGBoost'] = metrics['r2']
        self.mae_scores['XGBoost'] = metrics['mae']
        self.mape_scores['XGBoost'] = metrics['mape']
        return model

    def predicted_price(self):
        """
        Train an XGBoost model and add a predicted_price column to the DataFrame.
        """
        model = self.train_xgboost()  # Train the XGBoost model
        X_scaled_full = self.scaler.transform(self.df.drop(columns=['price']))
        self.df['predicted_price'] = model.predict(X_scaled_full)

    def save_predictions_to_csv(self, output_file):
        """
        Save the DataFrame with the predicted_price column to a CSV file.
        
        Parameters:
            output_file (str): Path to the output CSV file.
        """
        full_output_path = f"{self.base_path}\\{output_file}"
        self.df.to_csv(full_output_path, index=False)
        print(f"Predictions saved to {full_output_path}")

    def plot_price_vs_predicted(self):
        """
        Plot price vs. predicted_price.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['price'], self.df['predicted_price'], alpha=0.5, color='blue', label='Predicted vs. Actual')
        plt.plot([self.df['price'].min(), self.df['price'].max()], [self.df['price'].min(), self.df['price'].max()], color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual Price vs Predicted Price')
        plt.legend()
        plt.grid(True)
        plt_path = f"{self.base_path}\\price_vs_predicted_plot.png"
        plt.savefig(plt_path)
        plt.show()
        print(f"Plot saved to {plt_path}")

    def train_random_forest(self):
        """
        Train a Random Forest model and evaluate its performance.
        """
        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.calculate_metrics(self.y_test, y_pred)
        self.models['Random Forest'] = model
        self.r2_scores['Random Forest'] = metrics['r2']
        self.mae_scores['Random Forest'] = metrics['mae']
        self.mape_scores['Random Forest'] = metrics['mape']

    def train_svm(self):
        """
        Train a Support Vector Machine (SVM) model with hyperparameter tuning using GridSearchCV.
        """
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        }
        svm_model = GridSearchCV(SVR(), param_grid_svm, cv=5, scoring='r2')
        svm_model.fit(self.X_train, self.y_train)
        y_pred_svm = svm_model.predict(self.X_test)
        metrics_svm = self.calculate_metrics(self.y_test, y_pred_svm)
        self.models['SVM'] = svm_model.best_estimator_
        self.r2_scores['SVM'] = metrics_svm['r2']
        self.mae_scores['SVM'] = metrics_svm['mae']
        self.mape_scores['SVM'] = metrics_svm['mape']

    def train_custom_loss_gradient_boosting(self, learning_rate=0.1, n_estimators=15, max_depth=3):
        """
        Train a Gradient Boosting model and evaluate its performance.
        
        Parameters:
            learning_rate (float): Learning rate for the gradient boosting model.
            n_estimators (int): Number of boosting stages to be run.
            max_depth (int): Maximum depth of the individual trees.
        """
        model = GradientBoostingRegressor(
            learning_rate=learning_rate, 
            n_estimators=n_estimators, 
            max_depth=max_depth
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.calculate_metrics(self.y_test, y_pred)
        self.r2_scores['Gradient Boosting'] = metrics['r2']
        self.mae_scores['Gradient Boosting'] = metrics['mae']
        self.mape_scores['Gradient Boosting'] = metrics['mape']
      
    def train_stacking_regressor(self):
        """
        Train a Stacking Regressor using previously trained models and evaluate its performance.
        """
        # Train base models
        self.train_random_forest()
        self.train_linear_regression()
        self.train_svm()
        
        # Define estimators for stacking
        estimators = [
            ('rf', self.models['Random Forest']), 
            ('lr', self.models['Linear Regression']),
            ('svm', self.models['SVM'])
        ]
        final_estimator = Ridge(alpha=1.0)
        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        stacking_regressor.fit(self.X_train, self.y_train)
        y_pred_stack = stacking_regressor.predict(self.X_test)
        metrics_stack = self.calculate_metrics(self.y_test, y_pred_stack)
        self.r2_scores['Stacking Regressor'] = metrics_stack['r2']
        self.mae_scores['Stacking Regressor'] = metrics_stack['mae']
        self.mape_scores['Stacking Regressor'] = metrics_stack['mape']
        print(f"\nStacking uses Random Forest, Linear Regression, and SVM")

    def plot_model_performance(self):
        """
        Plots the R-squared scores, MAE, and MAPE of all models.
        """
        fig, ax1 = plt.subplots(figsize=(6, 4)) 

        # Bar plot for R-squared scores
        ax1.bar(self.r2_scores.keys(), self.r2_scores.values(), color='skyblue', label='R-squared')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('R-squared Score', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Set the tick locations and labels
        ax1.set_xticks(range(len(self.r2_scores)))
        ax1.set_xticklabels(self.r2_scores.keys(), rotation=90)

        # Line plot for MAE and MAPE on the same plot
        ax2 = ax1.twinx()  
        ax2.plot(self.mae_scores.keys(), self.mae_scores.values(), color='orange', marker='o', linestyle='--', label='MAE')
        ax2.plot(self.mape_scores.keys(), self.mape_scores.values(), color='green', marker='o', linestyle='--', label='MAPE')
        ax2.set_ylabel('Error Metrics', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Add legend
        fig.tight_layout()
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title('Model Performance Comparison')
        plt_path = f"{self.base_path}\\model_performance_plot.png"
        plt.savefig(plt_path)
        plt.show()
        print(f"Plot saved to {plt_path}")

    def plot_mape_performance(self):
        """
        Plots the MAPE of all models using short forms for model names.
        """
        # Define short names for models
        short_names = {
            'Linear Regression': 'LR',
            'XGBoost': 'XGB',
            'Random Forest': 'RF',
            'SVM': 'SVM',
            'Gradient Boosting': 'GB',
            'Stacking Regressor': 'SR'
        }

        # Replace full names with short names in the MAPE scores dictionary
        short_mape_scores = {short_names.get(model_name, model_name): score for model_name, score in self.mape_scores.items()}

        plt.figure(figsize=(6, 4)) 
        plt.bar(short_mape_scores.keys(), short_mape_scores.values(), color='lightcoral')
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Percentage Error (MAPE)')
        plt.title('Model MAPE Comparison')

        # Rotate x-axis labels to vertical
        plt.xticks(rotation=90)

        plt_path = f"{self.base_path}\\mape_performance_plot.png"
        plt.savefig(plt_path)
        plt.show()
        print(f"Plot saved to {plt_path}")

    def display_results(self):
        """
        Displays a table of R-squared, MAE, and MAPE for all models.
        """
        results = {
            'Model': [],
            'R-squared': [],
            'MAE': [],
            'MAPE': []
        }
        
        for model_name in self.r2_scores.keys():
            results['Model'].append(model_name)
            results['R-squared'].append(f"{self.r2_scores[model_name]:.4f}")
            results['MAE'].append(f"{self.mae_scores[model_name]:,.2f}")
            results['MAPE'].append(f"{self.mape_scores[model_name]:,.2f}%")
        
        results_df = pd.DataFrame(results)
        print("\nModel Performance Metrics:")
        print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
        
   
    def plot_residuals(self):
        residuals = self.df['price'] - self.df['predicted_price']
        plt.figure(figsize=(4, 4))
        plt.scatter(self.df['predicted_price'], residuals, alpha=0.7, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted Price')
        plt.grid(True)
        plt.show()
        
    def plot_actual_vs_predicted_lines(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.df.index, self.df['price'], label='Actual Price', color='blue')
        plt.plot(self.df.index, self.df['predicted_price'], label='Predicted Price', color='orange')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.title('Actual Price vs. Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_qq_plot(self):
        residuals = self.df['price'] - self.df['predicted_price']
        plt.figure(figsize=(6, 4))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ Plot of Residuals')
        plt.grid(True)
        plt.show()


# Direct execution
if __name__ == "__main__":
    file_name = "london_housing_with_stations.csv"
    pipeline = ModelPipeline(file_name)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.predicted_price()
    pipeline.save_predictions_to_csv('london_housing_with_predictions.csv')
    pipeline.train_xgboost()
    pipeline.train_custom_loss_gradient_boosting()
    pipeline.train_svm()
    pipeline.train_linear_regression()
    pipeline.train_stacking_regressor()
    pipeline.plot_model_performance()  
    pipeline.plot_mape_performance()  
    pipeline.display_results()
    pipeline.plot_price_vs_predicted()
    pipeline.plot_residuals()
    pipeline.plot_actual_vs_predicted_lines()
    pipeline.plot_qq_plot()
      

