from src.model.traing import *



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