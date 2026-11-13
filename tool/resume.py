"""Current-format checkpoints, complete epoch-boundary resume and run locks."""

from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch
from .temporal import FORMAT


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format",
        "model",
        "model_config",
        "train_config",
        "optimizer",
        "epoch",
        "step",
        "history",
        "rng",
        "loader_rng",
        "early",
        "best",
        "best_epoch",
        "total_steps",
    }
    if (
        not isinstance(payload, dict)
        or payload.get("format") != FORMAT
        or not required <= payload.keys()
    ):
        raise ValueError(
            "Expected a complete masked cross-attention SR checkpoint; older weights are unsupported"
        )
    return payload


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume(saved, config, model_config):
    if saved["model_config"] != model_config:
        raise ValueError("Resume requires the same SR model configuration")
    for section in (
        "training",
        "loss",
        "schedule",
        "augmentation",
        "dataset",
        "loader",
        "validation",
        "reporting",
    ):
        if config[section] != saved["train_config"][section]:
            raise ValueError(f"Resume requires unchanged {section} settings")


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@contextmanager
def training_lock(output):
    import fcntl

    with (Path(output) / ".training.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This training output is already in use") from None
        yield
