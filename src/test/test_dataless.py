import pandas as pd
from src.analytics.dataless_utils import *

df = pd.read_csv('../data/Plasma_MED_DECO_last.csv', encoding = "utf-8")
df.columns = ['ID', 'AN', 'PN', 'PD', 'PY', 'AY', 'TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'APPL', 'DRWG', 'EMBED', 'METHEX', 'REFF', 'CLM', 'PA', 'IPC', 'CPC', 'EMBEDDING']

#df_covid['TEXT'] = df_covid['TIEN'] + '.  '+ df_covid['ABEN'] + '.  '+ df_covid['CLMEN']
df_train = df[['TECHF']] #df[['TI', 'AB', 'TECHF']]
#drop rows with Nan Value
df_train = df_train.fillna('')
# drop row with empty string
df_train = df_train[df_train['TECHF'] != '']

df_train['text'] = df_train['TECHF'] #+ '.  '+ df_train['AB'] + '.  '+ df_train['TI']
df_train.drop(['TECHF'], inplace=True, axis=1)

#df_train['query'] = 'plasma for cancer treatment'
#df_train = df_train[['query', 'text']]
#
#df_train.to_csv("ABEN.csv",  encoding='utf-8', index=False)

trained_df = create_training_data(df_train)
#trained_df.to_csv("data_train_filter.csv", sep='\t', encoding='utf-8')
#df_p.to_csv("data_train_props.csv", sep='\t', encoding='utf-8')

#
trained_df.rename(columns={"label": "snorkel_label"}, inplace=True)

df_train_plasma_med = trained_df[trained_df.snorkel_label == 0]
df_train_plasma_deco = trained_df[trained_df.snorkel_label == 1]

df_train_plasma_med.to_csv("plasma/data_train_plasma_med_snorkel.csv", encoding='utf-8', index=False)
df_train_plasma_deco.to_csv("plasma/data_train_plasma_deco_snorkel.csv", encoding='utf-8', index=False)


