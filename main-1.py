import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

# Load the data, use own file path if necessary
# we used a data folder
data = pd.read_csv("data/train.csv")
X_train = data.drop('target', axis=1)
y_train = data['target'].copy()
X_test = pd.read_csv("data/test.csv")

# Intialize the stratified split
X_train, X_validation, y_train, y_validation = \
    train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Define the continuous variables for standardization
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

# Make preprocessor, standardize only continuous var, one hot encoded var are passed through as is
preprocessor = make_column_transformer(
        (StandardScaler(), continuous_variables),
        ('passthrough', categorical_variables))

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_validation = preprocessor.transform(X_validation)
X_test = preprocessor.transform(X_test)

# Define hyperparameter range
parameter_alt = {
    'n_estimators': [175, 500, 800, 1000],
    'min_samples_split': [1, 2, 3],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features' : ['sqrt', 'log2'],
    'class_weight' : ['balanced', 'balanced_subsample']
}

# Save optimal hyperparameters
parameter_opt = {
    'n_estimators': [800],
    'min_samples_split': [2],
    'min_samples_leaf': [3],
    'max_features' : ['log2'],
    'class_weight' : ['balanced_subsample']
}

# Define gridsearch and fit
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=parameter_alt, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search = GridSearchCV(estimator=rf, param_grid=parameter_opt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best parameter, estimation and acc (from gridsearch)
grid_search_params = grid_search.best_params_
grid_search_estimator = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

np.set_printoptions(threshold=np.inf)

# print grid results
print(f'Best Accuracy GRIDSEARCH: {best_accuracy:.6f}')
print("Best Parameters:", grid_search_params)
y_hat = grid_search_estimator.predict(X_validation)

# print out of sample performance
print("OUT-OF-SAMPLE prediction: ")
print(y_hat)

accuracy = accuracy_score(y_validation, y_hat)
print(f'OUT-OF-SAMPLE Accuracy: {accuracy}')

# Initialize RF with optimal parameters
print("ONLY RF")
rf_opt = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=800,min_samples_split=2,
                            min_samples_leaf=3, max_features='log2', class_weight='balanced_subsample')

# Reload data to train without the split
data = pd.read_csv("data/train.csv")
X_train = data.drop('target', axis=1)
y_train = data['target'].copy()
X_train = preprocessor.fit_transform(X_train)

# fit and predict
rf_opt.fit(X_train, y_train)
y_test = rf_opt.predict(X_test)

# Investigate some properties of predicted set, and put in bitstring
print(y_test)
ones = sum(y_test) / 5323
print("sum: " + str(sum(y_test)))
print("percentage 1: " + str(ones))

bitstring = ''.join(map(str, y_test))
print(bitstring)
