import sys
import pandas as pd
import datetime
import csv

filename = '/home/gustavo/Desktop/Mestrado/mestrado_dados/dados_18.11/cdr/cdr_regiao_imediata_jf_all_unique_filtered.txt'

df_info = pd.read_csv('/home/gustavo/Desktop/Mestrado/mestrado_dados/Regiao_Presumida/residence_antenna_jf.csv', sep=';')
df_info.rename(columns={'user': 'user_from'}, inplace=True)

df_ibge = pd.read_csv('/home/gustavo/Desktop/Mestrado/mestrado_dados/ibge/Sector_Class_Jf.csv')
df_ibge.drop(columns={'Unnamed: 0'}, inplace=True)
df_ibge = df_ibge[['V005', 'closest_antenna']]
df_ibge = df_ibge.groupby('closest_antenna').mean().reset_index()

def classe(i):
    if i <= 510:
        return 1
    elif i <= 1020:
        return 2
    elif i <= 1530:
        return 3
    elif i <= 2550:
        return 4
    elif i <= 5100:
        return 5
    elif i <= 10200:
        return 6
    elif i > 10200:
        return 7

df_ibge['classe'] = df_ibge['V005'].map(classe)
df_ibge = df_ibge.sort_values(['classe'])
df_ibge.rename(columns={'closest_antenna': 'residence_antenna'}, inplace=True)

array_df = []
for df_cdr in pd.read_csv(filename, sep=';', iterator=True, chunksize=1000000):
    array_df.append(df_cdr[['date', 'time','user_from', 'antenna']])

for df in array_df:
    df = pd.merge(df, df_info, on='user_from')

for df in array_df:
    df.rename(columns={'antenna':'residence_antenna'}, inplace=True)
    df = pd.merge(df, df_ibge[['residence_antenna', 'classe']], on='residence_antenna')

df_all = pd.concat(array_df)
print(len(df_all))
#df_all.to_csv('/home/gustavo/Desktop/Mestrado/mestrado_dados/dados_18.11/df_main_full_jf.csv', index=None, sep=';')
