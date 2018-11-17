import process_data
import pandas as pd 

df = process_data.load_data('./disaster_messages.csv', './disaster_categories.csv')
print(df.columns)

df = process_data.clean_data(df)
print(df)