import pandas as pd
import numpy as np
import acquire

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

def drop_cols(df, col_list):
    df = df.drop(columns=[col_list])

# prep the iris data
def split_iris_dataset(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.species)
    return train, validate, test

def prep_iris(df):
    # Drop the species_id and measurement_id columns
    df = df.drop(columns=['species_id', 'measurement_id'])
    
    # Rename the species_name column to just species
    df = df.rename(columns={'species_name': 'species'})
    
    # encode the species column
    df_dummies = pd.get_dummies(df[['species']], drop_first=True)
    df = pd.concat([df, df_dummies], axis=1)
    
    # split the data
    train, validate, test = split_iris_dataset(df)
    
    return train, validate, test

# prep the titantic data
def split_titanic_dataset(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)
    return train, validate, test

def impute(train, validate, test, my_strategy, column_list):
    imputer = SimpleImputer(strategy=my_strategy)
    train[column_list] = imputer.fit_transform(train[column_list])
    validate[column_list] = imputer.transform(validate[column_list])
    test[column_list] = imputer.transform(test[column_list])
    return train, validate, test

def prep_titanic(df):
    # drop missing observations of embark town
    df = df[~df.embark_town.isnull()]
    
    # use the pd.get_dummies as in lesson to encode embark_town column
    df_dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, df_dummies], axis=1)
    
    # Drop the embarked and deck columns
    df = df.drop(columns=['embarked', 'deck', 'passenger_id', 'class', 'sex', 'embark_town'])
    
    # split the data
    train, validate, test = split_titanic_dataset(df)
    
    # handle missing ages
    train, validate, test = impute(train, validate, test, 'median', ['age'])

    return train, validate, test