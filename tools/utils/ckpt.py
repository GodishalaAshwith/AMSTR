import os

import torch

from tools.utils.logging import get_logger


def save_ckpt(
    model,
    cfg,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    metrics,
    is_best=False,
    logger=None,
    prefix=None,
):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """
    if logger is None:
        logger = get_logger()
    if prefix is None:
        if is_best:
            save_path = os.path.join(cfg["Global"]["output_dir"], "best.pth")
        else:
            save_path = os.path.join(cfg["Global"]["output_dir"], "latest.pth")
    else:
        save_path = os.path.join(cfg["Global"]["output_dir"], prefix + ".pth")
    state_dict = model.module.state_dict() if cfg["Global"]["distributed"] else model.state_dict()
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": state_dict,
        "optimizer": None if is_best else optimizer.state_dict(),
        "scheduler": None if is_best else lr_scheduler.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    torch.save(state, save_path)
    logger.info(f"save ckpt to {save_path}")


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    if logger is None:
        logger = get_logger()
    checkpoints = cfg["Global"].get("checkpoints")
    pretrained_model = cfg["Global"].get("pretrained_model")

    # Auto-resolve .pth extension if omitted
    if checkpoints and not os.path.exists(checkpoints):
        if os.path.exists(checkpoints + ".pth"):
            checkpoints = checkpoints + ".pth"

    status = {}
    if checkpoints:
        if not os.path.exists(checkpoints):
            logger.warning(
                f"Checkpoint file not found: {checkpoints}. "
                "Ensure Google Drive is mounted (`from google.colab import drive; drive.mount('/content/drive')`) "
                "and verify the file path!"
            )
            logger.info("train from scratch")
            return status

        checkpoint = torch.load(checkpoints, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"resume from checkpoint {checkpoints} (epoch {checkpoint['epoch']}, step {checkpoint.get('global_step', 0)})")

        status["global_step"] = checkpoint.get("global_step", 0)
        status["epoch"] = checkpoint.get("epoch", 0) + 1
        status["metrics"] = checkpoint.get("metrics", {})
        return status
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model, logger)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model, logger):
    checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    for name in model.state_dict().keys():
        if name not in checkpoint["state_dict"]:
            logger.info(f"{name} is not in pretrained model")

