#!/usr/bin/env python
# coding: utf-8

#Import all relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
import pickle

# Script that prepares data, predicts salaries, and exports results
class salary_prediction_model():
    
    # Read the 'model' file which was saved
    def __init__(self, model_file):
        self.xgb = pickle.load(open(model_file, 'rb'))
    
    # Takes data, prepares data, makes predictions from trained model, and exports results to csv file
    def export_predictions(self, data_file):

        # Load csv file
        df_pred_features = pd.read_csv(data_file)
    
        # Saves jobId column for output file
        df_pred_jobId = pd.DataFrame(df_pred_features['jobId'])
    
        # Prepares data to be fed into the model
        df_pred_categories = df_pred_features[['jobType', 'degree', 'major', 'industry']]
        df_pred_categories = pd.get_dummies(df_pred_categories, drop_first=True)
        df_pred_features = pd.concat([df_pred_categories, df_pred_features[['yearsExperience','milesFromMetropolis']]], axis=1)
        del df_pred_categories
    
        # Loads model from disk, predicts salaries, and exports results to .csv file
        df_pred = pd.DataFrame(self.xgb.predict(df_pred_features))
        df_pred.columns = ['salary']
        df_pred = pd.concat([df_pred_jobId,df_pred], axis=1)
        df_pred.to_csv('predicted_salaries.csv')
        del df_pred_jobId
        
        # Informs user that process is complete
        print("Predictions exported to .csv file.")
    
    # Plot feature importance of model and save figure to .jpg file
    def export_feature_importance(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        xgboost.plot_importance(self.xgb, height=0.6, ax=ax)
        fig.savefig('feature_importance.jpg')
    
        # Informs user that process is complete
        print("Feature importances exported to .jpg file.")