from src.model.setFit_train import *


train_dataset, eval_dataset, test_dataset = prepare_trainingset("few_shot_train.csv", "few_shot_validation.csv", "few_shot_test.csv")
model_trainer = train_model("sentence-transformers/paraphrase-mpnet-base-v2",
                            train_dataset=train_dataset,
                            eval_dataset = eval_dataset,
                            labels=["PLASMA", "NO_PLASMA"])
print(evaluate(model_trainer, test_dataset))