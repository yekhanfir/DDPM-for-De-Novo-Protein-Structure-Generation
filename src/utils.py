import torch
from torch.optim.lr_scheduler import OneCycleLR
import json
import pickle

def create_scheduler(config):
    if config['use_scheduler']:
        scheduler = OneCycleLR(
            optimizer=config["optimizer"],
            max_lr=config["max_lr"],
            epochs=config["epochs"],
            steps_per_epoch=config["steps_per_epoch"],
            final_div_factor=config["final_div_factor"]
        )
    else: 
        return None
    return scheduler

def backward_pass(**kwargs):
    batch_loss = kwargs.get('batch_loss')
    optimizer = kwargs.get('optimizer')

    batch_loss.backward()
    optimizer.step()

def backward_pass_with_scheduler(**kwargs):
    scheduler = kwargs.get('scheduler')
    backward_pass(**kwargs)
    scheduler.step()


def create_backward_fn(use_scheduler):
    return (
        backward_pass_with_scheduler 
        if use_scheduler
        else backward_pass
    )


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