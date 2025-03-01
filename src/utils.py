import torch
from torch.optim.lr_scheduler import OneCycleLR
import json
import pickle

def create_scheduler(config):
    scheduler = OneCycleLR(
        optimizer=config["optimizer"],
        max_lr=config["max_lr"],
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        final_div_factor=config["final_div_factor"]
    )
    return scheduler

def save_metrics(metrics_dict):
    with open("metrics_dict.json","w") as f:
        json.dump(metrics_dict, f)

def save_model(model):
    dict_to_save = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(dict_to_save, 'model_state_dict.pt')


def save_training_examples(training_examples):
    f = open("example_proteins.pkl","wb")
    pickle.dump(training_examples, f)
    f.close()

def log_training_out(model, training_examples, metrics_dict):
    save_metrics(metrics_dict)
    save_model(model)
    save_training_examples(training_examples)