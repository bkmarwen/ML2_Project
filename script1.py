import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import mlflow
import mlflow.sklearn

# Function for preprocessing, model training, and MLflow logging
def train_and_log_model(model, param_grid, model_name='Model', cv=5, output_report='classification_report.txt'):
    # Step 1: Split the data into training and test sets
    # fetch dataset
    polish_companies_bankruptcy = fetch_ucirepo(id=365)

    # data (as pandas dataframes)
    X = polish_companies_bankruptcy.data.features
    y = polish_companies_bankruptcy.data.targets

    df = pd.concat([X, y], axis=1)
    df.rename(columns={df.columns[-1]: 'bankrupt'}, inplace=True)

    df.drop_duplicates(inplace= True)

    X = df.drop(columns='bankrupt')
    y = df['bankrupt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Imputation
    #imputer = SimpleImputer(strategy='median')
    
    # Step 3: Feature Selection
    selected_features = ['A6', 'A9', 'A24', 'A27', 'A29', 'A34', 'A35', 'A39', 'A40', 'A44', 'A46', 'A56', 'A58', 'A61']
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Step 4: Oversampling the training data to handle class imbalance
    over_sampler = RandomOverSampler()
    X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

    # Step 5: Preprocessing Pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Step 6: Full Pipeline with Model
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', model)
    ])
    
    # Step 7: Grid Search for Hyperparameter Tuning
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        scoring='recall'  # Use recall as the scoring metric
    )

    # Step 8: Start MLflow run to track training
    with mlflow.start_run():
        # Log initial data
        mlflow.log_param("selected_features", selected_features)
        mlflow.log_param("model_name", model_name)
        
        # Train and tune the model using GridSearchCV
        grid_search.fit(X_train_over, y_train_over)
        
        # Best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)  # Log best hyperparameters
        
        # Predictions on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save classification report to file and log to MLflow
        with open(output_report, 'w') as report_file:
            report = classification_report(y_test, y_pred)
            report_file.write(report)
        mlflow.log_artifact(output_report)
        
        # Log the trained model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Print results
        print(f"Model: {model_name}")
        print(f"Best Hyperparameters: {best_params}")
        print(report)

    return best_model

# Example usage with Decision Tree model and parameter grid
decision_tree_params = {
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Assuming you have your data in X, y
trained_model = train_and_log_model(
    model=DecisionTreeClassifier(random_state=42),
    param_grid=decision_tree_params,
    model_name='Decision Tree'
)
