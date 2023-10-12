import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv("data/train.csv")
X_train = data.drop('target', axis=1)
y_train = data['target'].copy()
y_train = y_train.to_numpy().reshape(-1, 1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_validation = pd.DataFrame(X_validation)
X_train = pd.DataFrame(X_train)
X_validation = pd.DataFrame(X_validation)

continuous_variables = ['womsixpq',	'perscapq',	'as_comp1',	'mainrppq',
                            'roomsq', 'vehq', 'hlthinpq', 'bedroomq', 'vehql',
                            'as_comp5',	'footwrpq',	'pettoypq',	'lifinspq',
                            'metropolitan', 'fam_size', 'cashcopq', 'age_ref',
                            'tobaccpq', 'male_ref',	'vehinspq', 'ttotalp',
                            'gasmopq', 'as_comp2', 'perslt18', 'bathrmq', 'othentpq',
                            'othaplpq', 'fdmappq', 'vrntlopq', 'educapq', 'dmsxccpq',
                            'fdhomepq',	'readpq', 'fdawaypq', 'bbydaypq', 'tvrdiopq',
                            'utilpq', 'trntrppq', 'urban', 'alcbevpq', 'miscpq',
                            'predrgpq',	'medsuppq',	'as_comp3',	'grlfifpq', 'persot64', 'medsrvpq',
                            'feeadmpq',	'fdxmappq',	'trnothpq', 'otheqppq',	'boyfifpq',
                            'age2',	'othhexpq',	'mensixpq',	'chldrnpq',	'as_comp4',	'hh_cu_q']
categorical_variables=X_train.columns.difference(continuous_variables)
preprocessor = make_column_transformer(
        (StandardScaler(), continuous_variables),
        ('passthrough', categorical_variables))
X_train = preprocessor.fit_transform(X_train)
X_validation = preprocessor.transform(X_validation)

parameter = {
    'n_estimators': [90, 100, 110, 120, 165, 175, 185, 400, 500, 700, 800],
    'min_samples_split': [1, 2, 3],
    'min_samples_leaf': [None, 1, 2, 3, 4, 5, 6],
    'max_features' : ['sqrt','log2', None],
    'class_weight' : ['balanced','balanced_subsample']
}
rf = RandomForestClassifier(random_state=42, n_jobs=2)
grid_search = GridSearchCV(estimator=rf, param_grid=parameter, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search_params = grid_search.best_params_
grid_search_estimator = grid_search.best_estimator_
print("Best Parameters:", grid_search_params)
y_hat = grid_search_estimator.predict(X_validation)
accuracy = accuracy_score(y_validation, y_hat)
print(f'Accuracy: {accuracy}')