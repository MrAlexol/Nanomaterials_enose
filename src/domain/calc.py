import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.target_name = None
        self.best_score = None
        self.best_estimator_str = ""
    
    def set_model(self, model, feature_columns: list[str], target_column: str):
        self.model = model
        self.feature_names = feature_columns
        self.target_name = target_column

    def train_knn(self, df: pd.DataFrame, feature_columns: list[str], target_column: str):
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует поле с признаком: {col}")
        if target_column not in df.columns:
            raise ValueError(f"Отсутствует поле: {target_column}")

        x = df[feature_columns].select_dtypes(include=['number']).dropna()
        y = df[target_column].loc[x.index]

        if x.empty or y.empty:
            raise ValueError("Данные для обучения пусты или содержат пропущенные поля.")

        params = {'n_neighbors': range(2, 8), 'weights': ['uniform', 'distance']}
        grid_searcher = GridSearchCV(KNN(), param_grid=params, cv=5, scoring='accuracy', n_jobs=1, error_score='raise')
        grid_searcher.fit(x, y)

        self.model = grid_searcher.best_estimator_
        self.feature_names = feature_columns
        self.target_name = target_column
        self.best_score = round(grid_searcher.best_score_, 3)
        self.best_estimator_str = str(grid_searcher.best_estimator_)

        return self.model

    def train_svm(self, df: pd.DataFrame, feature_columns: list[str], target_column: str):
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует поле с признаком: {col}")
        if target_column not in df.columns:
            raise ValueError(f"Отсутствует поле: {target_column}")

        x = df[feature_columns].select_dtypes(include=['number']).dropna()
        y = df[target_column].loc[x.index]

        if x.empty or y.empty:
            raise ValueError("Данные для обучения пусты или содержат пропущенные поля.")

        params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'probability': [True]
        }

        grid_searcher = GridSearchCV(SVC(), param_grid=params, cv=5, scoring='accuracy', n_jobs=1, error_score='raise')
        grid_searcher.fit(x, y)

        self.model = grid_searcher.best_estimator_
        self.feature_names = feature_columns
        self.target_name = target_column
        self.best_score = round(grid_searcher.best_score_, 3)
        self.best_estimator_str = str(grid_searcher.best_estimator_)

        return self.model

    def train_decision_tree(self, df: pd.DataFrame, feature_columns: list[str], target_column: str):
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует поле с признаком: {col}")
        if target_column not in df.columns:
            raise ValueError(f"Отсутствует поле: {target_column}")

        x = df[feature_columns].select_dtypes(include=['number']).dropna()
        y = df[target_column].loc[x.index]

        if x.empty or y.empty:
            raise ValueError("Данные для обучения пусты или содержат пропущенные поля.")

        params = {
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy']
        }

        grid_searcher = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, scoring='accuracy', n_jobs=1)
        grid_searcher.fit(x, y)

        self.model = grid_searcher.best_estimator_
        self.feature_names = feature_columns
        self.target_name = target_column
        self.best_score = round(grid_searcher.best_score_, 3)
        self.best_estimator_str = str(grid_searcher.best_estimator_)

        return self.model
    
    def train_logistic_regression(self, df: pd.DataFrame, feature_columns: list[str], target_column: str):
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует поле с признаком: {col}")
        if target_column not in df.columns:
            raise ValueError(f"Отсутствует поле: {target_column}")

        x = df[feature_columns].select_dtypes(include=['number']).dropna()
        y = df[target_column].loc[x.index]

        if x.empty or y.empty:
            raise ValueError("Данные для обучения пусты или содержат пропущенные поля.")

        params = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }

        grid_searcher = GridSearchCV(
            LogisticRegression(max_iter=1000), 
            param_grid=params, 
            cv=5, 
            scoring='accuracy',
            n_jobs=1
        )
        grid_searcher.fit(x, y)

        self.model = grid_searcher.best_estimator_
        self.feature_names = feature_columns
        self.target_name = target_column
        self.best_score = round(grid_searcher.best_score_, 3)
        self.best_estimator_str = str(grid_searcher.best_estimator_)

        return self.model

    def classify_batch(self, df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Классификатор не был обучен")

        df.columns = df.columns.str.strip()

        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют необходимые поля признаков: {missing}")

        x = df[self.feature_names].apply(pd.to_numeric, errors='coerce')
        x = x.dropna()

        if x.empty:
            raise ValueError("В файле с данными нет строк, подходящих для классификации")

        preds = self.model.predict(x)
        probs = self.model.predict_proba(x)

        values, counts = np.unique(preds, return_counts=True)
        majority_class = values[np.argmax(counts)]
        avg_proba = np.round(np.mean(probs, axis=0), 3)

        return majority_class, avg_proba, preds, probs
