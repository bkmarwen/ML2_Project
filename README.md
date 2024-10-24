# Predicting Bankruptcy in Poland

## Introduction
The goal of this project is to develop a predictive model that identifies potentially bankrupt companies using the Polish companies bankruptcy dataset. This is a crucial task for regulatory agencies, investors, and financial institutions to mitigate potential financial losses. Our focus is on maximizing recall to ensure that we capture as many bankrupt companies as possible.

## Dataset Description
- The Polish companies bankruptcy dataset contains information on companies that have filed for bankruptcy in Poland.
- The dataset contains **43,405 instances** and **64 features**. The features represent financial ratios and other variables that are commonly used in bankruptcy prediction studies.
- The dataset was collected over a period of 10 years (1999-2009) and was published on the UCI Machine Learning Repository.
- The objective of this dataset is to build a predictive model that can identify companies that are at risk of bankruptcy.
- The data was collected from bankruptcy filings and is divided into two classes: companies that filed for bankruptcy and those that did not. The data set includes 43,405 instances and 64 features, all of which are numerical. The aim is to predict bankruptcy, i.e., to determine whether a company will file for bankruptcy based on its financial ratios.

## Exploratory Data Analysis (EDA)
During the exploratory data analysis (EDA) of the Polish companies bankruptcy dataset, several issues were identified:
- **Duplicate Rows**: There were duplicated rows in the dataset, which were subsequently removed to avoid any bias in the analysis.
- **Missing Values**: A number of features had missing values, which needed to be addressed.
- **Distribution Analysis**: The distribution of values for one of the features, profit on operating activities, was examined using a boxplot. The plot revealed a highly skewed distribution with several outliers. This suggests that some form of data transformation or feature engineering may be required to improve the performance of the model.
- **Class Imbalance**: Another key observation from the EDA was the class imbalance in the dataset. Most companies in the dataset were not bankrupt, indicating that the majority class is much larger than the minority class. This has important implications for the modeling stage, as it may lead to biased results. Techniques such as oversampling or undersampling can be used to address this issue.
- **Model Suitability**: Autocorrelation issues were identified, indicating that linear models may not be the best fit for this dataset. Tree-based models, such as random forests or decision trees, may be better suited to capturing the non-linear relationships between the features and the target variable.

Overall, the EDA provided valuable insights into the data and highlighted several issues that need to be addressed during the data preparation and modeling stages.

## Data Preprocessing
The preprocessing steps included:
- **Splitting the Data**: The dataset was divided into training (80%) and testing (20%) sets.
- **Imputing Missing Values**: Median values were used for imputation.
- **Scaling**: StandardScaler was used to standardize features.
- **Addressing Class Imbalance**: RandomOverSampler was applied to oversample the minority class.
- **Feature Selection**: Recursive Feature Elimination (RFE) was employed to identify the top features.

## Model Building
We built and evaluated three machine learning models:
1. **Decision Tree**
2. **Random Forest**
3. **Gradient Boosting**

Hyperparameter tuning was performed using GridSearchCV, focusing on maximizing recall.

## Model Comparison and Selection
The performance of the models was compared using recall as the primary metric:
- **Gradient Boosting**: Highest recall of **0.806**.
- **Random Forest**: Recall of **0.421**.
- **Decision Tree**: Recall of **0.468**.

Gradient Boosting was selected as the best model for predicting bankrupt companies.

## Conclusion
This project highlights the importance of addressing class imbalance and selecting appropriate machine learning algorithms and hyperparameters. Our analysis demonstrates that Gradient Boosting is the most effective algorithm for predicting bankrupt companies in this dataset.

## How to Run the Project
1. Clone the repository.
2. Ensure you have the required libraries installed. You can install them using:
   ```bash
   pip install -r requirements.txt
