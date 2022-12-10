import wandb
import os
import numpy as np


def init_wandb(cfg) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        model (Torch Model): Model for Training
        args (TrainOptions,optional): TrainOptions class (refer options/train_options.py). Defaults to None.
    """
    
    wandb.init(
        name=cfg.wandb_name,
        config=cfg,
        project=cfg.wandb_project,
        resume="allow",
        id=cfg.wandb_id
    )


def wandb_log_train(train_loss, lr, iter):
    wandb.log({
        'Loss': train_loss,
        'Learning Rate': lr,
    }, step=iter)
    
def wandb_log_val(val_metric, iter):
    wandb.log({'F1@50 Score': val_metric}, step=iter)
