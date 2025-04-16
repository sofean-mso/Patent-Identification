# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from datasets import load_dataset, dataset_dict
import pandas as pd
from src.analytics.preprocess import soft_text_cleaning, soft_text_cleaning_stream


def prepare_trainingset(train_file_path,
                        validation_file_path,
                        test_file_path):
    """
    :param train_file_path:
    :param validation_file_path:
    :param test_file_path:
    :return:
    """

    dataset = load_dataset("csv", data_files={"train": train_file_path,
                                              "test": test_file_path,
                                               "validation": validation_file_path})

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    eval_dataset = dataset["validation"]

    return train_dataset, eval_dataset, test_dataset


def clean_traing_set(inpu_file_path:str, output_file_Path):
    """
    Cleaning texts of training data
    :param inpu_file_path: csv file with 'text' column
    :param output_file_Path:
    :return:
    """
    df = pd.read_csv(inpu_file_path, encoding = "utf-8")
    df = soft_text_cleaning_stream(df)
    df.to_csv(output_file_Path, encoding='utf-8', index=False)
    print("Cleaning data task is finished")


def prepare_trainingset(dataset_file_path:str, test_size=0.30):
    """
    prepare the training dataset for the DL model
    :param dataset_file_path:
    :param test_size:
    :return:
    """
    dataset = load_dataset("csv", data_files=dataset_file_path)
    dataset = dataset['train'].train_test_split(test_size=test_size, shuffle=True)

    return dataset
