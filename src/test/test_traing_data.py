from src.model.dataset import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



def creat_dataset():
    df_plasam_med = pd.read_csv("../data/2024/training/cleaned/tmp/data_train_plasma_med_snorkel_cleaned.csv", encoding="utf-8")
    df_plasam_med = df_plasam_med[['text']]
    # print(df_plasam_med.head(5))

    df_plasam_deco = pd.read_csv("../data/2024/training/cleaned/tmp/data_train_plasma_deco_snorkel_cleaned.csv",
                                 encoding="utf-8")
    df_plasam_deco = df_plasam_deco[['text']]
    # print(df_plasam_deco.head(5))

    # combine Med and DECO
    df_plasma_med_deco = pd.concat([df_plasam_med, df_plasam_deco])
    print('plasma MED and DECO')
    print(df_plasma_med_deco.head(5))
    print(len(df_plasma_med_deco))

    df_plasam_Tech = pd.read_csv("../data/2024/training/cleaned/tmp/data_train_plasma_tech_snorkel_cleaned.csv",
                                 encoding="utf-8")
    df_plasam_Tech = df_plasam_Tech[['text']]
    print("Plasma-related technology")
    print(df_plasam_Tech.head(5))
    print(len(df_plasam_Tech))
    df_plasam_Tech = df_plasam_Tech.head(3500)


    df_plasma_positive = pd.concat([df_plasma_med_deco, df_plasam_Tech])
    df_plasma_positive = df_plasma_positive.drop_duplicates()
    print('final Plasma POS')
    print(len(df_plasma_positive))

    test_positive_df = df_plasam_Tech.tail(2000)
    test_positive_df["label"] = "PLASMA"
    # print(len(df_plasma_positive))

    # add the label
    df_plasma_positive["label"] = "PLASMA"
    print(len(df_plasma_positive))
    print(df_plasma_positive.head(10))

    df_no_plasma= pd.read_csv("../data/2024/training/cleaned/tmp/data_train_no_plasma_tech_snorkel_cleaned.csv",
                                     encoding="utf-8")
    df_no_plasma = df_no_plasma[['text']]
    print("No Plasma-related technology")
    print(df_no_plasma.head(5))
    print(len(df_no_plasma))

    # filter out the short texts from no_plasma data
    df_no_plasma = df_no_plasma[df_no_plasma['text'].str.len() >= 50]
    # print(len(df_plasma_negative))
    # print(df_no_plasam.head(50))

    test_negative_df = df_no_plasma.tail(400)
    test_negative_df["label"] = "NO_PLASMA"

    df_no_plasma = df_no_plasma.head(3300)
    df_no_plasma = df_no_plasma.drop_duplicates()
    print(len(df_no_plasma))

    #Plasma blood
    df_plasma_blood = pd.read_csv("../data/2024/training/cleaned/tmp/data_train_blood_plasma_snorkel_cleaned.csv",
                                     encoding="utf-8")
    df_plasma_blood = df_plasma_blood[['text']]
    print(' #### blood Plasma')
    print(df_plasma_blood.head())
    print(len(df_plasma_blood))

    #Negitive Plasma
    df_plasma_negative =  pd.concat([df_no_plasma, df_plasma_blood])

    # add the label
    df_plasma_negative["label"] = "NO_PLASMA"
    print('----- All Negative ---')
    print(len(df_plasma_negative))
    print(df_plasma_negative.head(10))

    # final training dataset
    final_training_data = pd.concat([df_plasma_positive, df_plasma_negative])
    print('Final dataset Pos and Neg')
    print(len(final_training_data))

    # Ordering randomally the documents in the dataset
    final_training_data = shuffle(final_training_data)
    final_training_data = shuffle(final_training_data)
    print(final_training_data.head(100))

    # create train and test dataset

    df1 = final_training_data.iloc[:500, :]
    df2 = final_training_data.iloc[500:1000, :]
    df3 = final_training_data.iloc[1000:, :]
    #print(len(df1))
    #print(len(df2))
    #print(len(df3))

    # df3.to_csv("../data/training/cleaned/few_shot/few_shot_train.csv", encoding='utf-8', index=False)
    # df1.to_csv("../data/training/cleaned/few_shot/few_shot_validation.csv", encoding='utf-8', index=False)
    # df2.to_csv("../data/training/cleaned/few_shot/few_shot_test.csv", encoding='utf-8', index=False)

    test_dataset = pd.concat([test_positive_df, test_negative_df])
    test_dataset = shuffle(test_dataset)
    #test_dataset.to_csv("../data/2024/training/cleaned/final/test_dataset.csv", encoding='utf-8', index=False)
    # write data into disk
    #final_training_data.to_csv("../data/2024/training/cleaned/tmp/all/plasma_training_dataset.csv", encoding='utf-8', index=False)


def clean_data():
    clean_traing_set("../data/2024/CANPATFULL/data_train_no_plasma_tech_snorkel.csv",
                     "../data/2024/CANPATFULL/cleaned/data_train_no_plasma_tech_snorkel_cleaned.csv")
    print('Cleaning is finished!!')

def get_techF(txt):
    txt_list= txt.split(".###")
    return txt_list[0]

def extract_TechF():
    df = pd.read_csv("../data/2024/training/data_train_no_plasma_tech_snorkel.csv", encoding="utf-8")
    df['text'] = df['text'].map(lambda line: get_techF(line))
    df.to_csv("../data/2024/training/data_train_no_plasma_tech_snorkel_techf.csv", encoding='utf-8', index=False)


def check_row_duplicates():
    df1 = pd.read_csv("../data/2024/training/cleaned/final/test_dataset.csv",
                                     encoding="utf-8")
    df2 = pd.read_csv("../data/2024/training/cleaned/final/plasma_training_dataset.csv",
                                     encoding="utf-8")
    df = pd.merge(df1, df2, indicator=True, how='outer')\
                    .query('_merge=="left_only"')\
                    .drop('_merge', axis=1)
    print(len(df))
    print(df.head())
    df.to_csv("../data/2024/training/cleaned/final/test_dataset_unique.csv", encoding='utf-8', index=False)

def get_avg_text_length():
    df = pd.read_csv("../data/training/cleaned/new/plasma_traing_data.csv",
                      encoding="utf-8")
    df['length'] = df['text'].apply(
        lambda row: min(len(row.split(" ")), len(row)) if isinstance(row, str) else None
    )
    print(df.count())
    print(df.head())
    print(df['length'].mean() )
    filtered_df = df.query('length <= 128')
    print(filtered_df.count())


def df_shuffle():
    df = pd.read_csv("../data/training/cleaned/new/tmp/plasma_traing_data_0_1_with_blood.csv",
                     encoding="utf-8")
    df = shuffle(df)
    df.to_csv("../data/training/cleaned/new/tmp/plasma_traing_data_0_1_with_blood__.csv", encoding='utf-8', index=False)


def spliting_train_validate_dataset():
    df = pd.read_csv("../data/2024/training/cleaned/tmp/all/plasma_training_dataset_0_1.csv",
                     encoding="utf-8")
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                       random_state=104,
                                       test_size=0.25,
                                       shuffle=True)

    train_df = pd.DataFrame(X_train)
    train_df.columns = ['text']
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test)
    test_df.columns = ['text']
    test_df['label'] = y_test

    train_df.to_csv("../data/2024/training/cleaned/tmp/all/Train_plasma_training_dataset_0_1.csv", encoding='utf-8', index=False)
    test_df.to_csv("../data/2024/training/cleaned/tmp/all/Validation_plasma_training_dataset_0_1.csv", encoding='utf-8',
                index=False)




def combine_dfs():
    df_1 = pd.read_csv("../data/2024/CANPATFULL/cleaned/data_train_plasma_tech_snorkel_cleaned.csv",
                     encoding="utf-8")
    print('Plasma pos', len(df_1))
    df_2 = pd.read_csv("../data/2024/CANPATFULL/cleaned/data_train_blood_plasma_snorkel_cleaned.csv",
                       encoding="utf-8")
    print('Plasma blodd', len(df_2))
    df_3 = pd.read_csv("../data/2024/CANPATFULL/cleaned/data_train_no_plasma_tech_snorkel_cleaned.csv",
                       encoding="utf-8")
    print('No Plasma ', len(df_3))



    df_1 = df_1[['text']]
    df_1 = df_1[df_1['text'].str.len() >= 50]
    df_1['label'] = "PLASMA"
    print(len(df_1))

    df_2 = df_2[['text']]
    df_2 = df_2[df_2['text'].str.len() >= 50]
    df_2['label'] = "NO_PLASMA"
    print(len(df_2))

    df_3 = df_3[['text']]
    df_3 = df_3.head(600)
    df_3 = df_3[df_3['text'].str.len() >= 50]
    df_3['label'] = "NO_PLASMA"
    print(len(df_3))

    final = pd.concat([df_1, df_2, df_3])
    print(len(final))

    final.to_csv("../data/2024/CANPATFULL/cleaned/all/plasma_test_dataset.csv", encoding='utf-8', index=False)










#combine_dfs()


#extract_TechF()
#extract_TechF()
#df_shuffle()
#clean_data()
#clean_data()
#creat_dataset()
#check_row_duplicates()
#get_avg_text_length()
#spliting_train_validate_dataset()

#df = pd.read_csv("plasma/blood/data_train_blood_plasma_snorkel_cleaned.csv", encoding="utf-8")

#df['label'] = "NO_PLASMA"
#df1 = df[['text', 'label']]
#print(df1.head())
#df1.to_csv("plasma/blood/data_train_blood_plasma_snorkel_cleaned_labeld.csv", encoding='utf-8', index=False)

#df = pd.read_csv("../data/2024/CANPATFULL/cleaned/all/test_dataset.csv",encoding="utf-8")
#df.to_csv("../data/2024/CANPATFULL/cleaned/all/plasma_test_dataset.csv", encoding='utf-8',
               # index=False)









