# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# Data Preprocessing Class
class DataPreprocessor:
    def __init__(self):
        pass

    def impute_missing_values(self, data, strategy='mean'):
        """
        Fill missing values in the dataset.
        :param data: pandas DataFrame
        :param strategy: str, imputation method ('mean', 'median')
        :return: pandas DataFrame
        """
        numeric_data = data.select_dtypes(include=['number'])
        categorical_data = data.select_dtypes(exclude=['number'])

        if strategy == 'mean':
            data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())
        elif strategy == 'median':
            data[numeric_data.columns] = numeric_data.fillna(numeric_data.median())
        else:
            raise ValueError("Unknown strategy. Use 'mean' or 'median'.")

        # impute categorical columns with mode
        for col in categorical_data.columns:
            mode = categorical_data[col].mode()[0]
            data[col] = categorical_data[col].fillna(mode)

        return data

    def remove_duplicates(self, data):
        """
        Remove duplicate rows from the dataset.
        :param data: pandas DataFrame
        :return: pandas DataFrame
        """
        return data.drop_duplicates()

    def normalize_data(self, data, method='minmax'):
        """
        Apply normalization to numerical features.
        :param data: pandas DataFrame
        :param method: str, normalization method ('minmax' or 'standard')
        :return: pandas DataFrame
        """
        numeric_cols = data.select_dtypes(include=['number']).columns
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid normalization method. Use 'minmax' or 'standard'.")

        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data

    def remove_redundant_features(self, data, threshold=0.9):
        """
        Remove redundant or highly correlated features.
        :param data: pandas DataFrame
        :param threshold: float, correlation threshold
        :return: pandas DataFrame
        """
        correlation_matrix = data.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
        return data.drop(columns=to_drop)


# Logistic Regression Model Class
class SimpleModel:
    def __init__(self, scale_data=False):
        """
        Initialize the SimpleModel class.
        :param scale_data: bool, whether to scale data using StandardScaler
        """
        self.scale_data = scale_data
        self.model = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)

    def preprocess_features(self, data):
        """
        Encodes categorical features using one-hot encoding.
        :param data: pandas DataFrame (features)
        :return: pandas DataFrame (encoded features)
        """
        for col in data.columns:
            if data[col].dtype == 'object':
                data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
                data.drop(col, axis=1, inplace=True)
        return data

    def train_and_evaluate(self, input_data, print_report=False):
        """
        Train and evaluate a logistic regression model.
        :param input_data: pandas DataFrame (assumes first column is the target variable)
        :param print_report: bool, whether to print the classification report
        """
        # drop any missing data
        input_data = input_data.dropna()

        # split into features and target
        target = input_data.iloc[:, 0]
        features = self.preprocess_features(input_data.iloc[:, 1:])

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

        # scale data
        if self.scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # train the logistic regression model
        self.model.fit(X_train, y_train)

        # predict and evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy: {accuracy:.4f}')

        if print_report:
            print('Classification Report:')
            print(classification_report(y_test, y_pred))

        return accuracy



