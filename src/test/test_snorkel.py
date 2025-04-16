import pandas as pd
from src.analytics.snorkel_utils import *

def test_creat_training_datasets():
    df = pd.read_csv('../data/2024/CANPATFULL/CANPATFULL_Plasma_TECH_with_segs.csv', encoding="utf-8")
    df.columns = ['ID', 'AN', 'PN', 'PD', 'PY', 'AY', 'TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'APPL', 'DRWG', 'EMBED',
                  'METHEX', 'REFF', 'CLM', 'PA', 'IPC', 'CPC', 'EMBEDDING']

    # blood_plasma data
    #df1 = pd.read_csv('../data/2024/raw_data/Blood_Plasma_TECH_with_segs.csv', encoding="utf-8")
    #df1.columns = ['ID', 'AN', 'PN', 'PD', 'PY', 'AY', 'TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'APPL', 'DRWG', 'EMBED',
                  # 'METHEX', 'REFF', 'CLM', 'PA', 'IPC', 'CPC', 'EMBEDDING']

    #df3 = pd.concat([df, df1])
    #df_train = df3[['TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'EMBED', 'CLM']]
    df_train = df[['TECHF']]
    # drop rows with Nan Value
    df_train = df_train.fillna('')
    # drop row with empty string
    df_train = df_train[df_train['TECHF'] != '']

    df_train['text'] = df_train['TECHF'] #+ '.### ' + df_train['TI'] + '.###  ' + df_train['AB'] + '.###  ' +\
                      # df_train['BCKG']+ '.###  ' +df_train['SUMM']+ '.###  ' +df_train['EMBED']+ '.###  ' +df_train['CLM']
    # df_train.drop(['TECHF'], inplace=True, axis=1)
    #df_train.drop(['TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'EMBED', 'CLM'], inplace=True, axis=1)
    df_train.drop(['TECHF'], inplace=True, axis=1)

    # df_train['query'] = 'plasma for cancer treatment'
    # df_train = df_train[['query', 'text']]
    #
    # df_train.to_csv("ABEN.csv",  encoding='utf-8', index=False)

    trained_df = create_training_dataset(df_train)
    # trained_df.to_csv("data_train_filter.csv", sep='\t', encoding='utf-8')
    # df_p.to_csv("data_train_props.csv", sep='\t', encoding='utf-8')

    #
    trained_df.rename(columns={"label": "snorkel_label"}, inplace=True)
    df_train_plasma = trained_df[trained_df.snorkel_label == 0]
    df_train_no_plasma = trained_df[trained_df.snorkel_label == 1]
    df_train_neural = trained_df[trained_df.snorkel_label == -1]
    df_train_blood_plasma = trained_df[trained_df.snorkel_label == 2]

    df_train_plasma.to_csv("../data/2024/CANPATFULL/data_train_plasma_tech_snorkel.csv", encoding='utf-8', index=False)
    df_train_no_plasma.to_csv("../data/2024/CANPATFULL/data_train_no_plasma_tech_snorkel.csv", encoding='utf-8', index=False)
    df_train_neural.to_csv("../data/2024/CANPATFULL/data_train_neural_snorkel.csv", encoding='utf-8', index=False)
    df_train_blood_plasma.to_csv("../data/2024/CANPATFULL/data_train_blood_plasma_snorkel.csv", encoding='utf-8', index=False)



#test_creat_training_datasets()


