import re

import torch


def load_checkpoint_finetune(model, path):
    """Load backbone weights while dropping the original classifier head."""

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    filtered_state_dict = {
        key: value for key, value in state_dict.items() if not re.match(r"cls.*", key)
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    return model
