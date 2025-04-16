# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import csv
import json

import pandas as pd
from src.utils.postgres_utils import PostgresUtils


def extract_data_from_postgres(output_path:str):
    """

    :param output_path:
    :return:
    """
    #extract all document IDs
    postgres_util = PostgresUtils()
    doc_ids = postgres_util.get_doc_ids()
    data = set()
    counter = 0
    df = pd.DataFrame(columns=['ID', 'AN', 'PN', 'PD', 'PY', 'AY', 'TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'APPL',
                               'DRWG', 'EMBED', 'METHEX', 'REFF', 'CLM', 'PA', 'IPC', 'CPC', 'EMBEDDING'])
    for id in doc_ids:
        doc_data = postgres_util.get_doc_data_by_id(id)
        #doc_data_json = json.dumps(doc_data)
        ID = doc_data['ID']
        AN = doc_data['AN']
        PN = doc_data['PN']
        PD = doc_data['PD']
        PY = doc_data['PY']
        AY = doc_data['AY']
        TI = doc_data['TIEN']
        AB = doc_data['ABEN'].replace('\n', '\\n') if doc_data['ABEN'] else None
        TECHF = doc_data['TECHF'].replace('\n', '\\n') if doc_data['TECHF'] else None
        BCKG = doc_data['BCKG'].replace('\n', '\\n') if doc_data['BCKG'] else None
        SUMM = doc_data['SUMM'].replace('\n', '\\n') if doc_data['SUMM'] else None
        APPL = doc_data['APPL'].replace('\n', '\\n') if doc_data['APPL'] else None
        DRWG = doc_data['DRWG'].replace('\n', '\\n') if doc_data['DRWG'] else None
        EMBED = doc_data['EMBED'].replace('\n', '\\n') if doc_data['EMBED'] else None
        METHEX = doc_data['METHEX'].replace('\n', '\\n') if doc_data['METHEX'] else None
        REFF = doc_data['REFF'].replace('\n', '\\n') if doc_data['REFF'] else None
        CLM = ' \n '.join(doc_data['CLMEN']).replace('\n', '\\n') if doc_data['CLMEN'] else None
        PA = doc_data['PA']
        IPC = doc_data['IPC']
        CPC = doc_data['CPC']
        EMBEDDING = doc_data['EMBEDDING']

        #add new row into DF
        data = {'ID': ID, 'AN': AN, 'PN': PN, 'PD': PD, 'PY':PY, 'AY':AY, 'TI': TI, 'AB': AB, 'TECHF': TECHF, 'BCKG': BCKG,
                'SUMM': SUMM, 'APPL': APPL, 'DRWG': DRWG, 'EMBED': EMBED, 'METHEX': METHEX, 'REFF': REFF,
                'CLM': CLM, 'PA': PA, 'IPC': IPC, 'CPC': CPC, 'EMBEDDING': EMBEDDING}

        df = df._append(data, ignore_index=True)
        counter +=1
        if counter >= 300:
            write_df_into_disk(df, output_path)
            #remove all data in df
            df.drop(df.index, inplace=True)
            counter = 0


def write_df_into_disk(df, file_name):
    df.to_csv(file_name, mode='a', header=False, encoding='utf-8', index=False)
    print(len(df), ' records have been written into disk')


def extract_segment_text(seg_name:str, csv_file_path:str):
    df = pd.read_csv(csv_file_path, encoding="utf-8")
    df.columns = ['ID', 'AN', 'PN', 'PD', 'PY', 'AY', 'TI', 'AB', 'TECHF', 'BCKG', 'SUMM', 'APPL', 'DRWG', 'EMBED',
                  'METHEX', 'REFF', 'CLM', 'PA', 'IPC', 'CPC', 'EMBEDDING']

    # df_covid['TEXT'] = df_covid['TIEN'] + '.  '+ df_covid['ABEN'] + '.  '+ df_covid['CLMEN']
    seg_df = df[[seg_name]]
    return seg_df

#extract_data_from_postgres('jsonoutput_blood_plasma.csv')
