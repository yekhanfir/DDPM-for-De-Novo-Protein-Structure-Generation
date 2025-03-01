from dataclasses import dataclass

@dataclass
class DataConfig:
    data_file_path: str
    splits_file_path: str
    max_seq_len: str

@dataclass
class ModelConfig:
    timespteps: int
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
    
@dataclass
class GlobalConfig:
    training_config: TrainingConfig
    scheduler_config: SchedulerConfig
    data_config: DataConfig
    model_config: ModelConfig