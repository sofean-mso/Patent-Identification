# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import warnings

warnings.filterwarnings("ignore")

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from transformers import DataCollatorWithPadding
from datasets import Dataset
from src.model.eval_utils import compute_metrics
from src.model.dataset import prepare_trainingset

import pandas as pd
from evaluate import evaluator

import evaluate

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
import numpy as np

tokenizer = None


# Tokenize helper function
def tokenize(batch):
    """
    tokenize the dataset
    :param batch:
    :return:
    """
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)


def get_tokenizer(model_path):
    """
    Get the Bert Tokenizer instance
    :param model_path:
    :return:
    """
    bert_tokenizer = AutoTokenizer.from_pretrained(model_path)

    return bert_tokenizer


def train_classifier(model_path: str,
                     dataset,
                     output_dir="output",
                     train_batch_size=16,
                     eval_batch_size=8,
                     learning_rate=5e-7,  # 1.25e-5
                     num_epochs=10,
                     metric_for_best_model="accuracy"
                     ):
    """
    Train a Sequence classifier
    :param model_path:
    :param dataset:
    :param output_dir:
    :param train_batch_size:
    :param eval_batch_size:
    :param learning_rate:
    :param num_epochs:
    :param metric_for_best_model:
    :return:
    """
    dataset = dataset.rename_column("label", "labels")  # to match Trainer
    print(dataset)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(tokenized_dataset["train"].features.keys())

    # Prepare model labels - useful for inference
    num_labels = 2
    id2label = {0: "PLASMA", 1: "NO_PLASMA"}
    label2id = {"PLASMA": 0, "NO_PLASMA": 1}

    # Fine-tune & evaluate
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.25
    )

    print(" ############ Model Summary ######")
    print(model.cuda())
    print(" ############ End Summary ######")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        warmup_steps=0,
        num_train_epochs=num_epochs,
        torch_compile=True,  # optimizations
        optim="adamw_torch",  # improved optimizer
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.00,  # prevent overfitting default 0.01
        # fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model, trainer


def save_model(model_dir_path: str, trainer, tokenizer):
    """
    Write the model into disk
    :param model_dir_path:
    :param trainer:
    :param tokenizer:
    :return:
    """
    trainer.save_model(model_dir_path)
    tokenizer.save_pretrained(model_dir_path)
    print('Model is saved ..')


def evaluate_model(test_data_path, model_path):
    """
    Evaluate the model by test dataset
    :param test_data_path:
    :param model_path:
    :return:
    """
    pipe = pipeline(
        "text-classification", model=model_path, max_length=128
    )

    # Define dataset
    test_data = pd.read_csv(test_data_path)
    test_dataset = Dataset.from_pandas(test_data)

    # Define evaluator
    accuracy = evaluate.load("accuracy")

    # Evaluate accuracy
    eval = evaluator("text-classification")
    acc_result = eval.compute(
        model_or_pipeline=pipe,
        data=test_dataset,
        metric=accuracy,
        label_mapping={"PLASMA": 0, "NO_PLASMA": 1},
        strategy="bootstrap",
        n_resamples=100,
    )

    # Evaluate F1 score
    f1_metric = evaluate.load("f1")
    f1_result = eval.compute(
        model_or_pipeline=pipe,
        data=test_dataset,
        metric=f1_metric,
        label_mapping={"PLASMA": 0, "NO_PLASMA": 1},
        strategy="bootstrap",
        n_resamples=100,
    )
    print("########## Evaluation Results #########")
    print(" #### acc ###")
    print(acc_result)
    print(" #### f1 ###")
    print(f1_result)

    return acc_result, f1_result




if __name__ == '__main__':
    model_path = "allenai/scibert_scivocab_uncased"  # "anferico/bert-for-patents"
    tokenizer = get_tokenizer(model_path)

    dataset = prepare_trainingset('plasma_training_dataset_0_1.csv')
    model, trainer = train_classifier(model_path, dataset, num_epochs=10)
    # save the model
    save_model("plasma_model", trainer, tokenizer)

    # Evalaute the model with test dataset
    acc, f1 = evaluate_model("plasmatest.csv", "plasma_model")
    print(acc, f1)
