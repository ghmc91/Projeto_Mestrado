import sys
import pandas as pd
import datetime
import csv

filename = '/home/gustavo/Desktop/Mestrado/mestrado_dados/dados_18.11/df_main_full_jf.csv'
array_df = []
for df_cdr in pd.read_csv(filename, sep=';', iterator=True, chunksize=1000000):
    array_df.append(df_cdr)

df_all = pd.concat(array_df)
print(df_all['classe'].unique() , ' - ' , len(df_all))
