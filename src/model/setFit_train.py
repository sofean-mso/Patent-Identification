# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, TrainingArguments, sample_dataset
from setfit import Trainer
from datasets import load_dataset

import torch
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_trainingset(train_file_path,
                        validation_file_path,
                        test_file_path):
    """

    :param train_file_path:
    :param validation_file_path:
    :param test_file_path:
    :return:
    """
    #dataset = load_dataset("csv", data_files=file_path)
    dataset = load_dataset("csv", data_files={"train": train_file_path,
                                              "test": test_file_path,
                                               "validation": validation_file_path})

    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=600)  # dataset["train"]
    test_dataset = dataset["test"]
    eval_dataset = dataset["validation"].select(range(200))  # dataset["validation"]
    return train_dataset, eval_dataset, test_dataset


def train_model(model_name:str,
                train_dataset,
                eval_dataset,
                labels):
    """

    :param model_name:
    :param train_dataset:
    :param eval_dataset:
    :param labels:
    :return:
    """

    model = SetFitModel.from_pretrained(model_name,
                                        local_files_only=True,
                                        labels=labels)

    args = TrainingArguments(
        output_dir="setfit_output",
        batch_size=4,
        num_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        #learning_rate=5e-5, default
        num_iterations=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )

    trainer.train()

    return trainer


def evaluate(model_trainer, test_dataset):
    """

    :param model_trainer:
    :param test_dataset:
    :return:
    """
    metrics = model_trainer.evaluate(test_dataset)
    return metrics





if __name__ == '__main__':
    train_dataset, eval_dataset, test_dataset = prepare_trainingset("data/few_shot_train.csv",
                                                                    "data/few_shot_validation.csv",
                                                                    "data/few_shot_test.csv")

    model_trainer = train_model("sentence-transformers/paraphrase-mpnet-base-v2",
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                labels=["PLASMA", "NO_PLASMA"])
    print(evaluate(model_trainer, test_dataset))





