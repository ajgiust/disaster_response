import sys
import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine 
import os 


def load_data(messages_filepath, categories_filepath):
  '''Loads messages and categories datasets into program and merges datasets 
  together
  
  Arguments:
    messages_filepath {str} -- csv file path
    categories_filepath {str} -- csv file path
  '''

  # Load datasets
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)

  # Merge datasets
  return pd.merge(messages, categories, on='id')


def clean_data(df):
  '''Performs cleaning steps on data
  
  Arguments:
    df {dataframe} -- dataframe from load_data function
  '''

  ## 1. Split categories into separate category columns ##
  # Create a dataframe of the 36 individual category columns
  categories = pd.DataFrame(df['categories'].str.split(';', expand=True))

  # Select the first row of the categories dataframe
  row = categories[0:1]

  # Extract a list of new column names for categories
  category_colnames = row.apply(lambda x: x[0][:-2])

  # Rename the columns of `categories`
  categories.columns = category_colnames


  ## 2. Convert category values to just numbers 0 or 1 ##
  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')


  ## 3. Replace categories column in df with new category columns ##
  # drop the original categories column from `df`
  df.drop('categories', axis=1, inplace=True)


  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df,categories], axis=1)


  ## 4. Remove duplicates ##
  # drop duplicates
  df.drop_duplicates(inplace=True)

  return df



def save_data(df, database_filename):
  '''Saves dataframe into a SQL database 
  
  Arguments:
    df {dataframe} -- dataframe after being loaded and cleaned
    database_filename {str} -- path to [database filename] with '.db' extension
  '''
  engine = create_engine('sqlite:///' + database_filename)
  df.to_sql(database_filename, engine, index=False)  


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