from dataclasses import dataclass

@dataclass
class DataConfig:
    train_data_path: str
    validation_data_path: str
    hf_data_path: str
    max_seq_len: str

@dataclass
class ModelConfig:
    timesteps: int
    beta_start: float
    beta_end: float

@dataclass
class TrainingConfig:
    lr: float
    epochs: int
    batch_size: int
    shuffle_dataloader: bool

@dataclass
class SchedulerConfig:
    use_scheduler: bool
    final_div_factor: float
    div_factor: float
    max_lr: float
    
@dataclass
class GlobalConfig:
    training_config: TrainingConfig
    scheduler_config: SchedulerConfig
    data_config: DataConfig
    model_config: ModelConfig