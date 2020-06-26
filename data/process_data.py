import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, how='left', on=['id'])


def clean_data(df):
    
    #creating dataframe with 36 individual columns for each of the categories
    categories = df.categories.str.split(";", expand=True)
    
    #selecting first row to get data for processing in order to change naming of the columns
    row = categories.loc[[0]]
    
    #creating colum names
    category_columns = [row[x].str.split('-')[0][0] for x in row]
    categories.columns = category_columns #setting proper column names
    
    categories.related = categories.related.apply(lambda x: 'related-1' if 'related-2' in x esle x)
    
    #converting strings into numbers (the values looks like 'aid-1' so it makes is just 1') and the convert into integers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
        
    df.drop(columns=['categories'], inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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