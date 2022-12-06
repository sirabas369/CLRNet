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
        name=cfg.WANDB.NAME,
        config=cfg,
        project=cfg.WANDB.PROJECT,
        resume="allow",
        id=cfg.WANDB.RESTORE_NAME
    )


def wandb_log(train_loss, lr, iter):
    wandb.log({
        'Loss': train_loss,
        'Learning Rate': lr,
    }, step=iter)