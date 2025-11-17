import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from itertools import combinations
from collections import Counter

def preprocess_data(df, imputation_values, train_cols, scaler):
    """
    Preprocesses the raw test data using saved artifacts from training.
    """
    df_processed = df.copy()

    df_processed['Work_Experience_Is_Missing'] = df_processed['Work_Experience'].isnull().astype(int)

    for col, val in imputation_values.items():
        df_processed[col].fillna(val, inplace=True)

    spending_map = {'Low': 0, 'Average': 1, 'High': 2}
    df_processed['Spending_Score'] = df_processed['Spending_Score'].map(spending_map)

    nominal_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Var_1']
    df_processed = pd.get_dummies(df_processed, columns=nominal_cols, drop_first=True)

    current_cols = df_processed.columns

    missing_cols = set(train_cols) - set(current_cols)
    for c in missing_cols:
        df_processed[c] = 0

    df_processed = df_processed[train_cols]

    numerical_features_to_scale = ['Age', 'Work_Experience', 'Family_Size']
    df_processed[numerical_features_to_scale] = scaler.transform(df_processed[numerical_features_to_scale])

    return df_processed

def predict_ovo(classifiers, X_new):
    """
    Makes predictions using the One-vs-One strategy.
    """
    class_pairs = list(classifiers.keys())
    all_votes = []
    votes_by_classifier = {pair: clf.predict(X_new) for pair, clf in classifiers.items()}

    for i in range(X_new.shape[0]):
        votes_for_sample = [votes[i] for votes in votes_by_classifier.values()]
        final_prediction = Counter(votes_for_sample).most_common(1)[0][0]
        all_votes.append(final_prediction)

    return all_votes

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 12_classifier_ovo.py <path_to_test_file>")
        sys.exit(1)

    test_file_path = sys.argv[1]

    try:
        imputation_values = joblib.load('imputation_values.joblib')
        train_cols = joblib.load('train_cols.joblib')
        scaler = joblib.load('scaler.joblib')
        ovo_classifiers = joblib.load('ovo_classifiers.joblib')
    except FileNotFoundError:
        print("Error: One or more required .joblib files not found.")
        print("Please run the training notebook first to generate these files.")
        sys.exit(1)

    try:
        df_test = pd.read_csv(test_file_path)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file_path}")
        sys.exit(1)

    X_test_processed = preprocess_data(df_test, imputation_values, train_cols, scaler)

    predictions = predict_ovo(ovo_classifiers, X_test_processed) 

    output_df = pd.DataFrame({'predicted': predictions})
    output_df.to_csv('ovo.csv', index=False)
    print("Predictions successfully saved to ovo.csv")