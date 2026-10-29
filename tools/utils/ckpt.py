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
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
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
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Architectural migration: map legacy POPatchEmbed downsampling keys to SEAugmentedStem
        adapted_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if ".patch_embed.0." in k:
                new_k = k.replace(".patch_embed.0.", ".downsample_convs.0.")
            elif ".patch_embed.1." in k:
                new_k = k.replace(".patch_embed.1.", ".downsample_convs.1.")
            adapted_state_dict[new_k] = v

        try:
            model.load_state_dict(adapted_state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(
                f"Strict checkpoint loading encountered key differences ({e}). "
                "Gracefully falling back to strict=False for architectural migration (Tier 1 injection)..."
            )
            model.load_state_dict(adapted_state_dict, strict=False)

        if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state ({e}). Initializing clean optimizer state for new architecture.")
        if lr_scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                logger.warning(f"Could not restore scheduler state ({e}). Initializing clean scheduler for fine-tuning.")
        epoch_val = checkpoint.get('epoch', 1) if isinstance(checkpoint, dict) else 1
        step_val = checkpoint.get('global_step', 0) if isinstance(checkpoint, dict) else 0
        logger.info(f"resume from checkpoint {checkpoints} (epoch {epoch_val}, step {step_val})")

        status["global_step"] = step_val
        status["epoch"] = epoch_val
        status["metrics"] = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        return status
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model, logger)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model, logger):
    checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    adapted_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if ".patch_embed.0." in k:
            new_k = k.replace(".patch_embed.0.", ".downsample_convs.0.")
        elif ".patch_embed.1." in k:
            new_k = k.replace(".patch_embed.1.", ".downsample_convs.1.")
        adapted_state_dict[new_k] = v
    model.load_state_dict(adapted_state_dict, strict=False)
    for name in model.state_dict().keys():
        if name not in adapted_state_dict:
            logger.info(f"{name} is not in pretrained model (initialized fresh)")

