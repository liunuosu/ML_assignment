import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def dataloader(validation_size=0.1, random_state=42):
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")

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

    x_train = data_train.drop('target', axis=1)
    y_train = data_train['target'].copy()
    # 20.5 percent are 1 in y_train
    # 79.5 percent are 0 in y_train

    categorical_variables=x_train.columns.difference(continuous_variables)



    #print(y_train)
    x_test = data_test
    y_train = y_train.to_numpy().reshape(-1, 1)

    #encoder = OneHotEncoder()
    #y_train = encoder.fit_transform(y_train)
    #y_train = y_train.toarray()

    x_train, x_validation, y_train, y_validation = \
        train_test_split(x_train, y_train, test_size=validation_size, random_state=random_state,
                         stratify=y_train)

    # stratified k fold

    # TRY STRATIFIED sampling?
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_validation.shape)
    #print(y_validation.shape)
    # print(x_test.shape)
    # print(y_train[0:100])

    preprocessor = make_column_transformer(
        (StandardScaler(), continuous_variables),
        (OneHotEncoder(handle_unknown='ignore'), categorical_variables)
    )

    x_train = pd.DataFrame(x_train)
    x_validation = pd.DataFrame(x_validation)
    x_test = pd.DataFrame(x_test)
    print(x_train)
    print(x_validation)
    print(x_test)
    print(x_train.columns)

    x_train = preprocessor.fit_transform(x_train)
    x_validation = preprocessor.transform(x_validation)
    x_test = preprocessor.transform(x_test)

    print(x_train)
    print(x_train.shape)
    print(x_validation)

    return x_train, y_train, x_validation, y_validation, x_test
