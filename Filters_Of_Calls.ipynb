import csv 
import pandas as pd
import random

def filters_of_call(file_input, file_output):
    """
    Nessa função filtramos os registros pelo número de ligações
    e pela quantidade de dias em que o usuário realizou ligações
    """
    cdr = pd.read_csv(file_input, delimiter=';')
    cdr = cdr[cdr['duration'].between(0.07,120)]
    cdr_filter_by_quantity_of_call = cdr.groupby('user_from').size().reset_index().rename(columns={0:'QUANTITY_OF_CALLS'})

    cdr_filter_by_quantity_of_call = cdr_filter_by_quantity_of_call.loc[(
                                    (cdr_filter_by_quantity_of_call['QUANTITY_OF_CALLS'] >= 3) 
                                    & (cdr_filter_by_quantity_of_call['QUANTITY_OF_CALLS'] <= 500))]
    
    cdr_filter_by_quantity_of_days = cdr.groupby('user_from').date.nunique().to_frame('QUANTITY_OF_DAYS').reset_index()
    cdr_filter_by_quantity_of_days = cdr_filter_by_quantity_of_days.loc[(cdr_filter_by_quantity_of_days['QUANTITY_OF_DAYS'] >= 3)]
    
    df_merge = pd.merge(cdr_filter_by_quantity_of_call, cdr_filter_by_quantity_of_days,
                                    how='inner',on=['user_from'])
    cdr_after_filters = cdr[(cdr['user_from'].isin(df_merge['user_from']))]
    cdr_after_filters.to_csv(file_output)
    
    return cdr_after_filters
