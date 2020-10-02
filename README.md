# Salary Prediction Project (Python)

### **Define the Problem** 

**Project Goal:** The goal of this project is to analyze a dataset of job listings with posted salaries and predict the salaries of job listings that neglected to post a salary. 

**Use Case:** This type of model could be utilized by either recruiting firms or individual companies attempting to remain competitive in the job market.

**Tools:** The problem has been approached using Python 3 with machine learning algorithms/estimators from sklearn and xgboost.

### Description of Data

**The datasets given are as follows:**

train_features.csv: This file contains one million observations, each one representing an individual job listing, along with 8 features. One of these features, "jobId", is a unique identifier and was not utilized in the models construction for prediction. The remaining 7 features are described below.

train_salaries.csv: This file contains the salaries for the observations in train_features.csv along with their corresponding identifier.

test_features.csv: Silimar to train_features, however, an accompanying "test_salaries" csv is not provided.

**Features**

* companyId (categorical) - A company identifier (not utilized in final model)
* jobType (categorical) - Junior, Manager, Vice President, CEO, etc.
* degree (categorical) - High School, Bachelors, Masters, etc.
* major (categorical) - Chemistry, Math, Physics, etc.
* industry (categorical) - Auto, Health, Finance, Oil, etc.
* yearsExperience (numerical) - Desired years of experience of candidate
* milesFromMetropolis (numerical) - Job location's distance from metropolis

### Steps of Data Exploration

1. Import relevant libraries
    * pandas, numpy, matplotlib, seaborn, sklearn, xgboost, and pickle
2. Load the data into pandas DataFrames
3. Examine the structure of the data sets
4. Clean the data
    * Merge the features and salaries DataFrames on 'jobId' for easier handling
    * Check for missing values
    * Check for duplicate observations
    * Check for invalid data - rows with salaries equal to 0 are dropped
5. Vizualize the data (shown in Data Vizualization section below)
    * Bar plots for counts
    * Box and whisker plots for each category by salary
    * Line graphs for numerical features vs salary
6. Examine correlation between features/features and features/target (shown in Data Vizualization section below)
    * Heatmap correlation matrix
7. Run baseline model
    * A simple multiple linear regression model provided a MSE (mean-squared-error) score of ~925 through 5 cross-fold validation
8. Hypothesize feature engineering to improve accuracy over baseline model
    * Different types of encoding - One hot encoding showed most improvement
    * Binning degree, major, yearsExperience, and milesFromMetropolis features - These ideas did not show any improvement
9. Hypothesize different estimators to improve accuracy over baseline model
    * Stochastic Gradient Descent - Works well with large number of observations (>100K), may show improvement over linear regressor.
    * Decision Tree - The data is simple enough for binary splitting.
    * Random Forest - If moving forward with one-hot encoding and creating more variables, this algorithm could provide higher accuracy over decision tree.
    * Xgboost - Powerful boosting esemble that can optimize on least squares regression.

### Data Vizualization

**'jobType'**
![](images/JobType_Distribution.png)


### Model Development

### Model Deployment 