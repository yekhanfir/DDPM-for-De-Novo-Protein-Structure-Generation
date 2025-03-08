import os
import json
import pickle
import torch
from torch.optim.lr_scheduler import OneCycleLR

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

def save_metrics(metrics_dict, ouput_path):
    metrics_out_path = os.path.join(
        ouput_path, "metrics_dict.json"
    )
    with open(metrics_out_path,"w") as f:
        json.dump(metrics_dict, f)

def save_model(model, output_path):
    model_out_path = os.path.join(
        output_path, "model_state_dict.pt"
    )
    dict_to_save = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(dict_to_save, model_out_path)


def save_training_examples(training_examples, output_path):
    examples_out_path = os.path.join(
        output_path, "example_proteins.pkl"
    )
    f = open(examples_out_path,"wb")
    pickle.dump(training_examples, f)
    f.close()

def log_training_out(
        model, 
        training_examples, 
        metrics_dict, 
        output_path
    ):
    save_metrics(metrics_dict, output_path)
    save_model(model, output_path)
    save_training_examples(training_examples, output_path)