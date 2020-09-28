import pandas as pd
import numpy as np
import os
from env import host, user, password
import acquire
from sklearn.model_selection import train_test_split

#################### Acquire Mall Customers Data ##################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_telco2yr_data():
    '''This function reads n Telco data with the above query from Codeup database,
    writes it to a csv file, and returns the df'''
    sql_query = 'select customer_id, monthly_charges, tenure, total_charges from customers where contract_type_id = 3'
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco2yr.csv')
    return df

def get_telco2yr_data(cached=False):
    '''
    This function reads in Telco data with the above query from Codeup database if cached == False 
    or if cached == True reads in Telco data with the above query from a csv file, returns df
    '''
    if cached or os.path.isfile('telco2yr.csv') == False:
        df = new_telco2yr_data()
    else:
        df = pd.read_csv('telco2yr.csv', index_col=0)
    return df

def wrangle_telco():
    """
    This function takes acquired telco2yr data, completes the prep
    and splits the data into train, validate, and test datasets
    """
    df = get_telco2yr_data()
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    return train, validate, test

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

def inverse_scaled_columns(train, validate, test, scaler, columns_to_scale, columns_to_inverse):
    new_column_names = [c + '_inverse' for c in columns_to_inverse]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.inverse_transform(train[columns_to_inverse]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.inverse_transform(validate[columns_to_inverse]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.inverse_transform(test[columns_to_inverse]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test