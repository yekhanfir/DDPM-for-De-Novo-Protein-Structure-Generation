import torch
from model.DDPM import DDPM
from model.unet import UNet
from losses import masked_mse_loss

import hydra
import omegaconf
from config import (
    GlobalConfig, 
    TrainingConfig, 
    SchedulerConfig, 
    DataConfig,
    ModelConfig
)

from utils import create_scheduler, log_training_out
from data_handling.structure_dataset import DatasetFromDataframe

from tqdm import tqdm

@hydra.main(
    config_path="config_files/",
    version_base=None,
    config_name="unconditional_diffusion_config",
)
def get_config(cfg: omegaconf.DictConfig):
    config = GlobalConfig(
        training_config = TrainingConfig(
            **omegaconf.OmegaConf.to_container(cfg.training_config)
        ),
        scheduler_config = SchedulerConfig(
            **omegaconf.OmegaConf.to_container(cfg.scheduler_config)
        ),
        data_config = DataConfig(
            **omegaconf.OmegaConfig.to_container(cfg.data_config)
        ),
        model_config = ModelConfig(
            **omegaconf.omegaconf.to_container(cfg.model_config)
        )
    )
    return config

if __name__ == '__main__':
    config = get_config()

    train_set = DatasetFromDataframe(config.data_config)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.training_config.batch_size,
        shuffle=config.training_config.batch_size
    )
    train_iter = iter(train_loader)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM(
        unet=UNet(),
        device=device,
        model_config=config.model_config,
        max_seq_len=config.data_config.max_seq_len
    )

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(
            model, 
            device_ids=[
                device_id for device_id in range(torch.cuda.device_count())
            ]
        )
    model = model.to(device)
    
    lr=config.training_config.lr
    epochs=config.training_config.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = masked_mse_loss

    scheduler_config = {
        "optimizer": optimizer,
        "max_lr": lr,
        "epochs": epochs,
        "steps_per_epoch": len(train_loader),
        "max_div_factor": config.scheduler_config.max_div_factor
    }

    scheduler = create_scheduler(scheduler_config)

    example_proteins = {
        'original': [],
        'noisy': [],
        'predicted_noise': [],
        'timestep': []
    }
    metrics = {
        'step_wise_loss': [],
        'epoch_wise_loss': [],
    }

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for idx, batch in enumerate(iterator):
            x = batch['atom_positions'].to(device)

            # each mask is transformed into a stack of 3 copies if itself to match
            # the expected shape by the loss funtion
            mask = torch.stack([batch['atom_mask'] for _ in range(x.shape[-1])], dim=-1)
            mask = mask.to(device)

            # sample a batch of random time steps
            t = torch.randint(0, model.timesteps, [x.shape[0]], device=device, dtype=torch.long)

            # sample batch Gaussian noise
            noise = torch.randn(x.shape, device=device)

            # apply forward process
            noisy_x = model.forward_diffusion(x, t, noise)

            optimizer.zero_grad()

            # estimates the added noise at time step t using
            # the noise estimation networks
            predicted_noise = model.unet(noisy_x, t.unsqueeze(-1))

            loss = criterion(predicted_noise, noise, mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            iterator.set_postfix(loss=f"{loss.item():.4f}")
            metrics['step_wise_loss'].append(loss.item())

            if idx % (len(train_loader) // 5) == 0:
                # at each 20% of the epoch, saves the orinal and noisy backbones,
                # also saves the predicted noise and the timestep.
                # used to later investigate and evaluate model training.
                medium_noise_t = 0
                example_proteins['original'].append(x[medium_noise_t])
                example_proteins['noisy'].append(noisy_x[medium_noise_t])
                example_proteins['predicted_noise'].append(predicted_noise[medium_noise_t])
                example_proteins['timestep'].append(t[medium_noise_t])

        metrics['epoch_wise_loss'].append(total_loss/len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {metrics['epoch_wise_loss'][-1]:.4f}")

    metrics['epoch_wise_loss'] = [metrics['step_wise_loss'][0]] + metrics['epoch_wise_loss']
    
    log_training_out(model, example_proteins, metrics)