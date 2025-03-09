import os
import json
import pickle
import torch
from torch.optim.lr_scheduler import OneCycleLR

def create_scheduler(config):
    """Creates learning rate scheduler based on config.

    Args:
        config (dict): lr scheduler config.

    Returns:
        OneCycleLR: Scheduler object.
    """
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
    """
    Perform backprop and update gradients.
    """
    batch_loss = kwargs.get('batch_loss')
    optimizer = kwargs.get('optimizer')

    batch_loss.backward()
    optimizer.step()

def backward_pass_with_scheduler(**kwargs):
    """
    Performs backprop and updates gradients and learning rate.
    """
    scheduler = kwargs.get('scheduler')
    backward_pass(**kwargs)
    scheduler.step()

def create_backward_fn(use_scheduler):
    """Creates backward callable function.

    Args:
        use_scheduler (bool): whether or not scheduling is used.

    Returns:
        Callable: returns the right backward function.
    """
    return (
        backward_pass_with_scheduler 
        if use_scheduler
        else backward_pass
    )

def save_metrics(metrics_dict, ouput_path):
    """Saves training metrics to output path.

    Args:
        metrics_dict (dict): dictionary of logged metrics.
        ouput_path (str): path where to save metrics.
    """
    metrics_out_path = os.path.join(
        ouput_path, "metrics_dict.json"
    )
    with open(metrics_out_path,"w") as f:
        json.dump(metrics_dict, f)

def save_model(model, output_path):
    """Saves model checkpoint.

    Args:
        model (Any): model to be saved.
        output_path (str): path where to save model ckpt.
    """
    model_out_path = os.path.join(
        output_path, "model_state_dict.pt"
    )
    dict_to_save = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(dict_to_save, model_out_path)


def save_training_examples(training_examples, output_path):
    """Saves training inference examples, for later visualization.

    Args:
        training_examples (dict): dictionary containing original and denoised examples.
        output_path (str): path where training examples should be saved.
    """
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
    """Saves training experiment result.

    Args:
        model (Any): model to be saved.
        training_examples (dict): dictionary containing original and denoised examples.
        metrics_dict (dict): dictionary of logged metrics.
        output_path (Str): path where to save training experiment results.
    """
    save_metrics(metrics_dict, output_path)
    save_model(model, output_path)
    save_training_examples(training_examples, output_path)