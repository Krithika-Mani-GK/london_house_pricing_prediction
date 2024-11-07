The main dataset file is housing_london_dataset.csv 

There are two lookup files:
lkp_pc_loc.csv: Contains postcodes and corresponding locations used to fill in missing values in the main dataset.
lkp_pc_nearest_station.csv: Contains postcodes and their nearest stations. This file is used to add the nearest station attribute and calculate the distance between each postcode and its nearest station in miles.
lkp_crimerate_influence.csv: Contains the crime rate data for every location

During processing, two temporary files are used: 
temp_housing_london_loc.csv and temp_pc_ns_coordinates.csv. 

After thorough exploratory data analysis, the final preprocessed and feature-engineered dataset and output files are saved as london_housing_with_stations.csv and london_housing_with_stations_crimerate.csv and london_housing_with_predictions.csv

To execute the code, replace the path C:\Users\Keerthy M Ganesh\OneDrive - Loughborough University\Project_HousePricePrediction\Datasource with your own directory path (e.g., XXX/Datasource) in all relevant files. Update the path in HP_Menu.py for the image_url, in HP_Functions.py and HP_FeatureEngg.py for BASE_PATH, in HP_Preprocessing.py and HP_Accuracy.py for self.base_path, and in HP_SQL_DBValidation.py for BASE_PATH. Then, copy all files from the Datasource folder (inside 23COP328_F332804_Code_Submission_26082024) into your XXX/Datasource/ directory, ensuring the path updates reflect this location. After making these changes, you can execute the code as intended. 

Similarly, to view the Tableau file "HousePriceVisualisation.twb", edit the datasource and point to XXX/Datasource/ directory.

To view the WEKA files, install WEKA and open the arff files. The source files for WEKA are weka_london_housing_with_stations_crimerate.csv & weka_london_housing_with_stations.csv; The database file is house_price_prediction.db

I have recorded a video titled 23COP328_F332804_HousePricePrediction_Demonstration.mp4, which captures the entire demo. You can watch the video for a complete explanation of the demonstration.

The report is available as 23COP328_F332804_HousePricePrediction_Project_Report.pdf
