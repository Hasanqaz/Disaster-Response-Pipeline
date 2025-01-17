import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: messages_filepath: Path of messages data in CSV format, categories_filepath: Path of categories data in CSV format
    
    Output: df:  Merge both messages and categories into one dataframe
    '''
    
    # Read message data
    messages = pd.read_csv(messages_filepath)
    
    # Read categories data
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    
    '''
    Input: df: Merged dataset from messages and categories
    
    Output: df: Cleaned dataset
    '''
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories
    category_colnames = [s.strip('-0') for s in row]
    category_colnames[0] = category_colnames[0].replace('-1', '')
    
    # Rename the categories columns
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for i in categories.columns:
        for j in range(categories.shape[0]):
            if categories[i][j].find('1') != -1:
                categories[i][j] = 1
            else:
                categories[i][j] = 0
    
    # Drop the original categories column
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new categories dataframe
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)

    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='last')
    
    return df


def save_data(df, database_filename):
    
    '''
    Input: df: cleaned dataset, database_filename: database name
    
    Output: SQLite database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterMessages', engine, index=False)  


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