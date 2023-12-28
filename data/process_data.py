import sys
import pandas as pd
import re

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ input  : fielpaths of the messages and catagories csv's
        output : merged and partly cleaned dataframe with all data (df)"""
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # split entry data on ';'
    categories = categories['categories'].str.split(';', expand=True)

    # create columns names from entry data
    columns = [item.replace("0", "").replace("-","").replace("1","") for item in list(categories.iloc[0])]

    #rename dataframe columns
    categories.columns = columns

    # filter text data from data entries in dataframe
    categories = categories.applymap(lambda x: re.sub("[^0-9]", "", x))

     # merge data
    df = pd.concat([messages, categories], axis=1, join="inner")
    
    print(df)
    return df

def clean_data(df):
    """ input  : merged dataframe
        output : merged dataframe without duplicates"""
    
    # dropping duplicated
    df_no_duplicates = df.drop_duplicates()
    print(f'number of duplicates: {len(df)-len(df_no_duplicates)}')
    df = df_no_duplicates.copy()
    
    return df


def save_data(df, database_filepath):
    """ input : dataframe (df) and filepath for database (database_filepath)
        output : dataframe will be saved to defined file path"""
    print(f'database klaasje : {database_filepath}')
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('category_messages', engine, index=False, if_exists= 'replace')
    pass  

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print(f'hallo Nicolaas : database_filepath = {database_filepath}')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()