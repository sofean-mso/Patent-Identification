# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns


def evaluate(model_trainer, test_dataset):
    """
    :param model_trainer:
    :param test_dataset:
    :return:
    """
    metrics = model_trainer.evaluate(test_dataset)
    return metrics


def get_label(d):
  if d['label'] == 'PLASMA':
    return 0
  else:
    return 1


def plot_cm(cm):
    """

    :param cm:
    :return:
    """
    classes = ['PLASMA', 'NO_PLASMA']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sns.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    fig = ax.get_figure()
    fig.savefig("confusion_matrix.png")




def plot_confusion_matrix(model_path:str, dataset):
    """
    plot the confusion matrix for test data
    :param model_path:
    :param dataset:
    :return:
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    fined_model = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=128)

    predictions = fined_model(dataset['test']['text'])
    predictions = [get_label(d) for d in predictions]
    cm = confusion_matrix(dataset['test']['label'], predictions, normalize='true')
    plot_cm(cm)




def compute_metrics(pred):
    """
    Computes Accuracy, F1, precision, and recall for a given set of predictions.
    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids
    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)
    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)
    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


