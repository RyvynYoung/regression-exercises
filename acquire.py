
import pandas as pd
import env
import os

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# 1. get_titanic_data: returns the titanic data from the codeup data science database as a pandas data frame.
def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col=0)
        return df
    else:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        df.to_csv(filename)
        return df
    
# 2. get_iris_data: returns the data from the iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species in addition to the species_ids.
def get_iris_data():
    filename = 'iris.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col=0)
        return df
    else:
        df = pd.read_sql('select * from measurements join species using (species_id);', get_connection('iris_db'))
        df.to_csv(filename)
        return df


# 3. Once you've got your get_titanic_data and get_iris_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for a local filename like titanic.csv or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, 
# then write the dataframe to a .csv file with the appropriate name
